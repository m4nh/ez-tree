import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/Addons.js';
import appleRedUrl from '../../assets/apple_red.glb?url';

/** Built-in fruit models, loaded on first use */
export const FruitModels = {
  'Red Apple': appleRedUrl,
};

export const FruitAnchor = {
  Leaves: 'leaves',
  Twigs: 'twigs',
};

export class FruitOptions {
  enabled = true;
  model = 'Red Apple';

  // Placement
  anchor = FruitAnchor.Leaves;
  density = 0.12;
  maxCount = 150;
  seed = 0;
  spacing = 1.0;
  minHeight = 0.15;
  maxHeight = 1;
  twigRadius = 0.3;

  // Shape
  size = 1.2;
  sizeVariance = 0.2;
  gravity = 0.85;
  tilt = 15;
  outward = 0.3;
  hang = 0.1;

  // Color
  tint = 0xffffff;
  colorVariance = 0.15;

  // Motion
  sway = 4;
  swaySpeed = 1.5;
}

// Random values drawn per candidate anchor. A fixed count keeps the stream
// aligned, so moving a slider adds/removes fruits instead of reshuffling them.
const R_ACCEPT = 0;
const R_SIZE = 1;
const R_SPIN = 2;
const R_TILT = 3;
const R_TILT_AXIS = 4;
const R_HUE = 5;
const R_LIGHT = 6;
const R_SAT = 7;
const R_ALONG = 8;
const R_AROUND = 9;
const R_DELAY = 10;
const R_COUNT = 11;

const UP = new THREE.Vector3(0, 1, 0);
const DOWN = new THREE.Vector3(0, -1, 0);
const IDENTITY = new THREE.Quaternion();
const DEG = Math.PI / 180;

// Grow-in timing (seconds)
const GROW_DURATION = 0.45;
const GROW_STAGGER = 0.7;

// Shake timing (seconds)
const SHAKE_FALL_TIME = 3.6;
const SHAKE_SHRINK_TIME = 0.4;
const GRAVITY = 40;

const _m = new THREE.Matrix4();
const _s = new THREE.Vector3();
const _q = new THREE.Quaternion();
const _v = new THREE.Vector3();

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function easeOutBack(x) {
  const c1 = 1.70158;
  const c3 = c1 + 1;
  return 1 + c3 * Math.pow(x - 1, 3) + c1 * Math.pow(x - 1, 2);
}

const clamp01 = (x) => Math.min(1, Math.max(0, x));

/**
 * Scatters instanced fruit models over a Tree's skeleton.
 *
 * Fruits hang from leaf origins (or twig surfaces) read from
 * `tree.skeleton`, so no library changes are needed. The tree is watched for
 * regeneration and fruits re-scatter automatically. Uses its own RNG so the
 * tree's seed-reproducible growth is never disturbed.
 */
export class Fruits extends THREE.Group {
  /**
   * @param {import('@dgreenheck/ez-tree').Tree} tree
   * @param {FruitOptions} options
   */
  constructor(tree, options = new FruitOptions()) {
    super();
    this.name = 'Fruits';
    this.tree = tree;
    this.options = options;

    /** @type {Map<string, {name: string, parts: {geometry: THREE.BufferGeometry, material: THREE.Material, depthMaterial: THREE.Material}[], height: number, triangles: number}>} */
    this.models = new Map();
    this.model = null;
    this.meshes = [];
    this.capacity = 0;
    this.placements = [];
    this.anim = null;
    this.time = 0;
    this.onScatter = null;

    this.uniforms = {
      uFruitTime: { value: 0 },
      uFruitSway: { value: 0 },
      uFruitSwaySpeed: { value: 0 },
    };
    this.syncUniforms();

    this._skeleton = null;
    this._candidates = null;
    this._candidateKey = null;
    this._loadToken = 0;
    this._loader = new GLTFLoader();

    if (options.enabled) tree.add(this);
  }

  /** Number of fruits currently placed */
  get count() {
    return this.placements.length;
  }

  /** Triangles drawn across all fruit instances */
  get triangleCount() {
    return this.model ? this.model.triangles * this.count : 0;
  }

  /**
   * Loads a GLB/glTF from a URL or ArrayBuffer and registers it under `name`.
   * Returns the registered name (suffixed if it collides).
   * @param {string | ArrayBuffer} source
   * @param {string} name
   */
  async loadModel(source, name) {
    const gltf = typeof source === 'string'
      ? await this._loader.loadAsync(source)
      : await this._loader.parseAsync(source, '');

    let unique = name;
    for (let i = 2; this.models.has(unique); i++) unique = `${name} (${i})`;

    this.models.set(unique, this.#buildModel(gltf, unique));
    return unique;
  }

  /**
   * Switches to a registered (or built-in) model and pops the fruits in
   * @param {string} name
   */
  async useModel(name) {
    const token = ++this._loadToken;
    if (!this.models.has(name) && FruitModels[name]) {
      await this.loadModel(FruitModels[name], name);
    }
    if (token !== this._loadToken) return;

    const model = this.models.get(name);
    if (!model) throw new Error(`Unknown fruit model "${name}"`);

    this.options.model = name;
    this.model = model;
    this.#rebuildMeshes(Math.max(64, this.options.maxCount));
    this.scatter({ pop: true });
  }

  setEnabled(enabled) {
    this.options.enabled = enabled;
    this.#stopAnimation();
    if (enabled) {
      this.tree.add(this);
      this.scatter({ pop: true });
    } else {
      this.tree.remove(this);
      this.scatter();
    }
  }

  syncUniforms() {
    this.uniforms.uFruitSway.value = this.options.sway * DEG;
    this.uniforms.uFruitSwaySpeed.value = this.options.swaySpeed;
  }

  /**
   * Recomputes fruit placements from the tree skeleton and options.
   * @param {{pop?: boolean}} [opts] Animate the fruits growing in
   */
  scatter({ pop = false } = {}) {
    this._skeleton = this.tree.skeleton;
    this.#stopAnimation();

    const o = this.options;
    const placements = [];

    if (o.enabled && this.model) {
      const candidates = this.#candidates();

      let minY = Infinity;
      let maxY = -Infinity;
      for (const c of candidates) {
        minY = Math.min(minY, c.position.y);
        maxY = Math.max(maxY, c.position.y);
      }
      const span = maxY - minY || 1;

      const grid = new Map();
      const cell = Math.max(o.spacing, 1e-3);

      for (const c of candidates) {
        // Candidates are sorted by their acceptance roll
        if (c.r[R_ACCEPT] >= o.density || placements.length >= o.maxCount) break;

        const h = (c.position.y - minY) / span;
        if (h < o.minHeight || h > o.maxHeight) continue;

        const p = this.#place(c, h);
        if (o.spacing > 0 && this.#crowded(grid, cell, p.position, o.spacing)) continue;

        const key = this.#cellKey(p.position, cell);
        if (!grid.has(key)) grid.set(key, []);
        grid.get(key).push(p.position);
        placements.push(p);
      }
    }

    this.placements = placements;
    if (placements.length > this.capacity) {
      this.#rebuildMeshes(Math.max(placements.length, this.options.maxCount));
    }

    placements.forEach((p, i) => this.#writeInstance(i, p.position, p.quaternion, p.scale, p.color));
    this.#commit(placements.length);

    if (pop) this.pop();
    this.onScatter?.(this);
  }

  /** Grows every fruit in from zero, rippling from the bottom of the canopy up */
  pop() {
    if (!this.placements.length) return;
    this.anim = { type: 'grow', start: this.time };
    this.#animate();
  }

  /** Shakes the tree: fruits fall, bounce, settle, then regrow */
  shake() {
    if (!this.placements.length || !this.model) return;
    this.#stopAnimation();

    const rand = mulberry32(Math.random() * 2 ** 32);
    const bodies = this.placements.map((p) => {
      const axis = new THREE.Vector3(rand() - 0.5, rand() - 0.5, rand() - 0.5).normalize();
      return {
        position: p.position.clone(),
        quaternion: p.quaternion.clone(),
        velocity: new THREE.Vector3((rand() - 0.5) * 4, rand() * 3, (rand() - 0.5) * 4),
        spinAxis: axis,
        spinRate: (rand() * 2 - 1) * 10,
        rest: new THREE.Quaternion().setFromAxisAngle(UP, rand() * Math.PI * 2),
        restY: this.model.height * p.scale,
        // Lower fruits are knocked loose first
        delay: 0.1 + p.h * 0.35 + rand() * 0.5,
      };
    });

    this.anim = {
      type: 'shake',
      start: this.time,
      last: this.time,
      bodies,
      treeRotation: this.tree.rotation.clone(),
    };
  }

  /**
   * Per-frame update: wind sway, regeneration tracking and animations
   * @param {number} t Elapsed time in seconds
   */
  update(t) {
    this.time = t;
    this.uniforms.uFruitTime.value = t;

    if (this.tree.skeleton !== this._skeleton) {
      this.scatter();
    }

    if (this.anim) this.#animate();
  }

  /**
   * Builds a plain-mesh copy of the fruits for export: one node per fruit
   * sharing the model's geometry, so the file needs no EXT_mesh_gpu_instancing
   * support. Per-fruit colors are baked into materials, shared per color.
   * Uses the rest transforms, ignoring any running animation.
   */
  createExportGroup() {
    const group = new THREE.Group();
    group.name = 'Fruits';
    if (!this.model) return group;

    const materials = new Map();
    const exportMaterial = (material, color) => {
      const key = `${material.uuid}:${color.getHexString()}`;
      if (!materials.has(key)) {
        const m = material.clone();
        m.onBeforeCompile = () => {};
        m.color?.multiply(color);
        materials.set(key, m);
      }
      return materials.get(key);
    };

    this.placements.forEach((p, i) => {
      const fruit = new THREE.Group();
      fruit.name = `Fruit_${i}`;
      fruit.position.copy(p.position);
      fruit.quaternion.copy(p.quaternion);
      fruit.scale.setScalar(p.scale);

      for (const part of this.model.parts) {
        const material = Array.isArray(part.material)
          ? part.material.map((m) => exportMaterial(m, p.color))
          : exportMaterial(part.material, p.color);
        fruit.add(new THREE.Mesh(part.geometry, material));
      }
      group.add(fruit);
    });

    return group;
  }

  // --------------------------------------------------------------------------
  // Placement
  // --------------------------------------------------------------------------

  /**
   * Candidate anchors for the current skeleton/anchor mode, sorted by
   * acceptance roll. Cached until the inputs change.
   */
  #candidates() {
    const o = this.options;
    const skeleton = this.tree.skeleton;
    if (!skeleton) return [];

    const key = `${o.anchor}|${o.seed}|${o.anchor === FruitAnchor.Twigs ? o.twigRadius : ''}`;
    if (this._candidates && this._candidateSkeleton === skeleton && this._candidateKey === key) {
      return this._candidates;
    }

    const rand = mulberry32(Math.imul(o.seed + 1, 0x9e3779b1));
    const draw = () => {
      const r = new Float32Array(R_COUNT);
      for (let i = 0; i < R_COUNT; i++) r[i] = rand();
      return r;
    };

    const candidates = [];

    if (o.anchor === FruitAnchor.Twigs) {
      for (const branch of skeleton.branches) {
        const sections = branch.sections;
        for (let i = 0; i < sections.length - 1; i++) {
          // Always draw so eligibility changes don't shift the stream
          const r = draw();
          const a = sections[i];
          const b = sections[i + 1];
          const radius = THREE.MathUtils.lerp(a.radius, b.radius, r[R_ALONG]);
          if (radius > o.twigRadius) continue;

          const position = new THREE.Vector3().lerpVectors(a.origin, b.origin, r[R_ALONG]);
          const axis = UP.clone().applyEuler(a.orientation);
          const theta = r[R_AROUND] * Math.PI * 2;
          const side = new THREE.Vector3(Math.cos(theta), 0, Math.sin(theta));
          side.addScaledVector(axis, -side.dot(axis));
          if (side.lengthSq() < 1e-6) side.set(1, 0, 0);
          side.normalize();

          position.addScaledVector(side, radius);
          candidates.push({ position, direction: side, r });
        }
      }
    } else {
      for (const leaf of skeleton.leaves) {
        candidates.push({
          position: leaf.origin,
          direction: UP.clone().applyEuler(leaf.orientation),
          r: draw(),
        });
      }
    }

    candidates.sort((a, b) => a.r[R_ACCEPT] - b.r[R_ACCEPT]);

    this._candidates = candidates;
    this._candidateSkeleton = skeleton;
    this._candidateKey = key;
    return candidates;
  }

  /** Computes the transform and color of one fruit at a candidate anchor */
  #place(c, h) {
    const o = this.options;
    const r = c.r;

    // Hang along the anchor direction, pulled toward straight down by gravity
    const align = new THREE.Quaternion()
      .setFromUnitVectors(DOWN, c.direction)
      .slerp(IDENTITY, o.gravity);

    const spin = new THREE.Quaternion().setFromAxisAngle(UP, r[R_SPIN] * Math.PI * 2);

    const tiltAngle = r[R_TILT_AXIS] * Math.PI * 2;
    const tilt = new THREE.Quaternion().setFromAxisAngle(
      _v.set(Math.cos(tiltAngle), 0, Math.sin(tiltAngle)),
      (r[R_TILT] * 2 - 1) * o.tilt * DEG,
    );

    const quaternion = tilt.multiply(align).multiply(spin);

    const position = c.position.clone()
      .addScaledVector(c.direction, o.outward);
    position.y -= o.hang;

    const scale = Math.max(0.05, o.size * (1 + o.sizeVariance * (r[R_SIZE] * 2 - 1)));

    const v = o.colorVariance;
    const color = new THREE.Color(o.tint).offsetHSL(
      (r[R_HUE] - 0.5) * 0.12 * v,
      (r[R_SAT] - 0.5) * 0.4 * v,
      (r[R_LIGHT] - 0.5) * 0.5 * v,
    );

    return { position, quaternion, scale, color, h, delay: r[R_DELAY] };
  }

  #cellKey(p, cell) {
    return `${Math.floor(p.x / cell)},${Math.floor(p.y / cell)},${Math.floor(p.z / cell)}`;
  }

  #crowded(grid, cell, p, spacing) {
    const cx = Math.floor(p.x / cell);
    const cy = Math.floor(p.y / cell);
    const cz = Math.floor(p.z / cell);
    const s2 = spacing * spacing;
    for (let x = cx - 1; x <= cx + 1; x++) {
      for (let y = cy - 1; y <= cy + 1; y++) {
        for (let z = cz - 1; z <= cz + 1; z++) {
          const bucket = grid.get(`${x},${y},${z}`);
          if (bucket?.some((q) => q.distanceToSquared(p) < s2)) return true;
        }
      }
    }
    return false;
  }

  // --------------------------------------------------------------------------
  // Model & meshes
  // --------------------------------------------------------------------------

  /**
   * Bakes a glTF into normalized parts: largest dimension = 1 unit, pivot at
   * the top (stem) of the model so it hangs below its anchor.
   */
  #buildModel(gltf, name) {
    const root = gltf.scene;
    root.updateMatrixWorld(true);

    const parts = [];
    const box = new THREE.Box3();
    root.traverse((o) => {
      if (!o.isMesh) return;
      const geometry = o.geometry.clone().applyMatrix4(o.matrixWorld);
      geometry.computeBoundingBox();
      box.union(geometry.boundingBox);
      const material = Array.isArray(o.material)
        ? o.material.map((m) => m.clone())
        : o.material.clone();
      parts.push({ geometry, material });
    });

    if (!parts.length) throw new Error('The model contains no meshes');

    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z) || 1;

    // Pivot on the centroid of the topmost vertices, which is where the stem
    // usually is, rather than the bounding-box center
    const topY = box.max.y - size.y * 0.05;
    const pivot = new THREE.Vector3();
    let n = 0;
    for (const { geometry } of parts) {
      const pos = geometry.attributes.position;
      for (let i = 0; i < pos.count; i++) {
        if (pos.getY(i) >= topY) {
          pivot.x += pos.getX(i);
          pivot.z += pos.getZ(i);
          n++;
        }
      }
    }
    pivot.set(n ? pivot.x / n : 0, box.max.y, n ? pivot.z / n : 0);

    const normalize = new THREE.Matrix4()
      .makeScale(1 / maxDim, 1 / maxDim, 1 / maxDim)
      .multiply(new THREE.Matrix4().makeTranslation(-pivot.x, -pivot.y, -pivot.z));

    let triangles = 0;
    for (const part of parts) {
      part.geometry.applyMatrix4(normalize);
      part.geometry.computeBoundingSphere();
      triangles += (part.geometry.index?.count ?? part.geometry.attributes.position.count) / 3;

      const materials = Array.isArray(part.material) ? part.material : [part.material];
      materials.forEach((m) => this.#addSway(m));

      part.depthMaterial = new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking });
      this.#addSway(part.depthMaterial);
    }

    return { name, parts, height: size.y / maxDim, triangles };
  }

  /** Injects a pendulum sway around each fruit's stem into a material */
  #addSway(material) {
    material.onBeforeCompile = (shader) => {
      Object.assign(shader.uniforms, this.uniforms);

      shader.vertexShader = `
        uniform float uFruitTime;
        uniform float uFruitSway;
        uniform float uFruitSwaySpeed;

        mat3 fruitSwayRotation() {
          #ifdef USE_INSTANCING
            vec3 anchor = instanceMatrix[3].xyz;
          #else
            vec3 anchor = vec3(0.0);
          #endif
          float phase = dot(anchor, vec3(0.73, 0.21, 0.57));
          float t = uFruitTime * uFruitSwaySpeed;
          float ax = uFruitSway * sin(t + phase);
          float az = uFruitSway * 0.7 * sin(1.31 * t + 1.7 * phase);
          float cx = cos(ax), sx = sin(ax), cz = cos(az), sz = sin(az);
          return mat3(1.0, 0.0, 0.0, 0.0, cx, sx, 0.0, -sx, cx) *
                 mat3(cz, sz, 0.0, -sz, cz, 0.0, 0.0, 0.0, 1.0);
        }
      ` + shader.vertexShader;

      shader.vertexShader = shader.vertexShader
        .replace(
          '#include <beginnormal_vertex>',
          '#include <beginnormal_vertex>\nobjectNormal = fruitSwayRotation() * objectNormal;',
        )
        .replace(
          '#include <begin_vertex>',
          '#include <begin_vertex>\ntransformed = fruitSwayRotation() * transformed;',
        );
    };
    material.customProgramCacheKey = () => 'ez-tree-fruit-sway';
    material.needsUpdate = true;
  }

  #rebuildMeshes(capacity) {
    for (const mesh of this.meshes) {
      this.remove(mesh);
      mesh.dispose();
    }
    this.meshes = [];
    this.capacity = capacity;
    if (!this.model) return;

    this.model.parts.forEach((part, i) => {
      const mesh = new THREE.InstancedMesh(part.geometry, part.material, capacity);
      mesh.name = `${this.model.name}_${i}`;
      mesh.customDepthMaterial = part.depthMaterial;
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      // Instances move every frame while animating
      mesh.frustumCulled = false;
      // Allocate instance colors up front so the shader compiles with them
      mesh.setColorAt(0, new THREE.Color(1, 1, 1));
      mesh.count = 0;
      this.meshes.push(mesh);
      this.add(mesh);
    });
  }

  #writeInstance(i, position, quaternion, scale, color) {
    _m.compose(position, quaternion, _s.setScalar(Math.max(scale, 1e-4)));
    for (const mesh of this.meshes) {
      mesh.setMatrixAt(i, _m);
      if (color) mesh.setColorAt(i, color);
    }
  }

  #commit(count = this.placements.length) {
    for (const mesh of this.meshes) {
      mesh.count = count;
      // Hidden meshes are skipped by GLTFExporter, so empty ones don't export
      mesh.visible = count > 0;
      mesh.instanceMatrix.needsUpdate = true;
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    }
  }

  // --------------------------------------------------------------------------
  // Animation
  // --------------------------------------------------------------------------

  #stopAnimation() {
    if (this.anim?.type === 'shake') {
      this.tree.rotation.copy(this.anim.treeRotation);
    }
    if (this.anim) {
      this.anim = null;
      this.placements.forEach((p, i) => this.#writeInstance(i, p.position, p.quaternion, p.scale));
      this.#commit();
    }
  }

  #animate() {
    if (this.anim.type === 'grow') this.#animateGrow();
    else this.#animateShake();
    this.#commit();
  }

  #animateGrow() {
    const elapsed = this.time - this.anim.start;
    this.placements.forEach((p, i) => {
      const delay = GROW_STAGGER * (0.6 * p.h + 0.4 * p.delay);
      const k = clamp01((elapsed - delay) / GROW_DURATION);
      this.#writeInstance(i, p.position, p.quaternion, p.scale * easeOutBack(k));
    });
    if (elapsed > GROW_STAGGER + GROW_DURATION) this.#stopAnimation();
  }

  #animateShake() {
    const anim = this.anim;
    const elapsed = this.time - anim.start;
    const dt = Math.min(this.time - anim.last, 1 / 30);
    anim.last = this.time;

    // Damped wobble of the whole tree from its base
    const wobble = 0.03 * Math.exp(-3.5 * elapsed) * Math.sin(elapsed * 22);
    this.tree.rotation.set(
      anim.treeRotation.x + wobble * 0.6,
      anim.treeRotation.y,
      anim.treeRotation.z + wobble,
    );

    if (elapsed > SHAKE_FALL_TIME + SHAKE_SHRINK_TIME) {
      this.#stopAnimation();
      this.pop();
      return;
    }

    const shrink = 1 - clamp01((elapsed - SHAKE_FALL_TIME) / SHAKE_SHRINK_TIME);

    anim.bodies.forEach((b, i) => {
      const p = this.placements[i];
      if (elapsed >= b.delay) {
        b.velocity.y -= GRAVITY * dt;
        b.position.addScaledVector(b.velocity, dt);

        if (b.position.y <= b.restY) {
          b.position.y = b.restY;
          if (b.velocity.y < -3) {
            // Bounce, losing energy and spin
            b.velocity.y *= -0.35;
            b.velocity.x *= 0.7;
            b.velocity.z *= 0.7;
            b.spinRate *= 0.5;
          } else {
            b.velocity.y = 0;
          }
          const friction = Math.exp(-4 * dt);
          b.velocity.x *= friction;
          b.velocity.z *= friction;
          // Settle upright on the ground
          b.quaternion.slerp(b.rest, 1 - Math.exp(-6 * dt));
        } else {
          _q.setFromAxisAngle(b.spinAxis, b.spinRate * dt);
          b.quaternion.premultiply(_q);
        }
      }
      this.#writeInstance(i, b.position, b.quaternion, p.scale * shrink);
    });
  }
}
