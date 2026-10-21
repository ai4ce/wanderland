import { number, sceneLabel, formatBytes, loadJSON } from './explore.js';

const $ = (selector) => document.querySelector(selector);
const assetNames = { '3dgs': '3dgs.sog', pointcloud: 'pointcloud.glb', mesh: 'mesh.glb' };
const modalityLabels = { '3dgs': '3DGS', pointcloud: 'Point Cloud', mesh: 'Mesh' };

function viewerMessage(message, loading = false, percent) {
  $('#viewer-status').hidden = !message;
  $('#viewer-message').textContent = message;
  $('#viewer-progress').hidden = !loading;
  if (percent === undefined) $('#viewer-progress').removeAttribute('value');
  else $('#viewer-progress').value = percent;
  $('#viewer').setAttribute('aria-busy', String(loading));
}

function showDetails(data) {
  const counts = data.counts;
  const extent = data.bbox_m.max.map((max, axis) => number.format(max - data.bbox_m.min[axis])).join(' × ');
  const rows = [
    ['Fisheye images', number.format(counts.fisheye_images)],
    ['LiDAR points (raw)', number.format(counts.las_points)],
    ['LiDAR points (preview)', number.format(counts.preview_points)],
    ['Gaussians', number.format(counts.gaussians)],
    ['Mesh vertices / faces', `${number.format(counts.mesh_vertices)} / ${number.format(counts.mesh_faces)}`],
    ['NVS train / validation', `${number.format(counts.nvs_train)} / ${number.format(counts.nvs_val)}`],
    ['Camera poses', number.format(counts.camera_poses)],
    ['Extent (X × Y × Z)', `${extent} m`],
    ['Ground coverage', `${number.format(data.ground_coverage_m2)} m²`],
    ['Trajectory length', `${number.format(data.trajectory_length_m)} m`],
    ['Point preview voxel size', `${number.format(data.voxel_size_m * 100)} cm`],
    ['3D previews (total)', formatBytes(Object.values(data.assets).reduce((sum, file) => sum + file.bytes, 0))],
    ['Raw scene (all listed files)', formatBytes(Object.values(data.raw.files).reduce((sum, file) => sum + file.bytes, 0))],
  ];
  for (const [label, value] of rows) {
    const term = document.createElement('dt');
    const detail = document.createElement('dd');
    term.textContent = label;
    detail.textContent = value;
    $('#scene-stats').append(term, detail);
  }
  $('#raw-folder').href = data.raw.folder_url;
  for (const name of ['3dgs.ply', 'scene.ply', 'lidar/colorized.las', 'fisheye.tar.gz', 'transforms.json']) {
    const file = data.raw.files[name];
    if (!file) continue;
    const item = document.createElement('li');
    const link = document.createElement('a');
    link.href = file.url;
    link.textContent = name;
    const size = document.createElement('span');
    size.textContent = formatBytes(file.bytes);
    link.append(size);
    item.append(link);
    $('#download-links').append(item);
  }
  $('#preview-sizes').textContent = `Preview sizes: ${Object.entries(data.assets).map(([name, file]) => `${name} ${formatBytes(file.bytes)}`).join(' · ')}. Sizes use decimal units.`;
}

async function initViewer(data, assetBase) {
  // Gate before importing GPU libraries or creating a canvas on mobile.
  if (matchMedia('(pointer: coarse), (max-width: 767px)').matches) {
    viewerMessage('Open this scene on a desktop with a mouse or trackpad to explore in 3D. Scene details and downloads are available below.');
    return;
  }
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('webgl2', { antialias: false });
  if (!context) {
    viewerMessage('This browser does not support WebGL2. Try a desktop browser with hardware acceleration enabled.');
    return;
  }
  viewerMessage('Loading 3D viewer…', true);
  const [THREE, { SparkRenderer, SplatMesh, SparkControls }, { GLTFLoader }, { MeshoptDecoder }] = await Promise.all([
    import('three'),
    import('@sparkjsdev/spark'),
    import('three/addons/loaders/GLTFLoader.js'),
    import('three/addons/libs/meshopt_decoder.module.js'),
  ]);
  const renderer = new THREE.WebGLRenderer({ canvas, context, antialias: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
  renderer.setClearColor(0x10151c);
  const scene = new THREE.Scene();
  const level = new THREE.Group();
  scene.add(level);
  const camera = new THREE.PerspectiveCamera(60, 1, 0.05, 2000);
  camera.up.fromArray(data.world_up);
  const controls = new SparkControls({ canvas });
  controls.fpsMovement.moveSpeed = 2;
  controls.fpsMovement.enable = false;
  controls.pointerControls.moveInertia = 0;
  controls.pointerControls.rotateInertia = 0;
  const yawToUp = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), camera.up.clone().normalize());
  const levelFrame = new THREE.Quaternion();
  function horizontalFrame() {
    // Match Spark's YXZ heading, including when looking straight up or down.
    const angles = new THREE.Euler().setFromQuaternion(camera.quaternion.clone().premultiply(yawToUp.clone().invert()), 'YXZ');
    return new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), angles.y).premultiply(yawToUp);
  }
  function updateLevelReadout() {
    // Show the actual composed correction in the starting horizontal view frame.
    const rotation = levelFrame.clone().invert().multiply(level.quaternion).multiply(levelFrame);
    const angles = new THREE.Euler().setFromQuaternion(rotation, 'YXZ');
    const degrees = (angle) => {
      const value = Math.round(THREE.MathUtils.radToDeg(angle) * 10) / 10;
      return `${value > 0 ? '+' : value < 0 ? '−' : ''}${Math.abs(value).toFixed(1)}°`;
    };
    $('#level-readout').textContent = `Level: pitch ${degrees(-angles.x)} · roll ${degrees(angles.z)} · yaw ${degrees(angles.y)}`;
  }
  function adjustLevel(axis, degrees) {
    const rotation = new THREE.Quaternion().setFromAxisAngle(axis, THREE.MathUtils.degToRad(degrees));
    // Rotate every modality about the viewer, not the dataset's origin.
    level.position.sub(camera.position).applyQuaternion(rotation).add(camera.position);
    level.quaternion.premultiply(rotation).normalize();
    updateLevelReadout();
  }
  function resetView() {
    const pose = data.thumbnails[0].camera;
    camera.position.fromArray(pose.position);
    camera.quaternion.fromArray(pose.quaternion);
    camera.fov = pose.fov;
    camera.updateProjectionMatrix();
    level.position.set(0, 0, 0);
    level.quaternion.identity();
    levelFrame.copy(horizontalFrame());
    updateLevelReadout();
    controls.pointerControls.moveVelocity.set(0, 0, 0);
    controls.pointerControls.rotateVelocity.set(0, 0, 0);
    controls.pointerControls.scroll.set(0, 0, 0);
  }
  resetView();
  canvas.tabIndex = 0;
  canvas.setAttribute('aria-label', 'Interactive scene. Drag to look; WASD or arrow keys to move; I/K pitch, J/L yaw, U/O roll the level.');
  canvas.addEventListener('pointerdown', () => canvas.focus());
  canvas.addEventListener('focus', () => { controls.fpsMovement.enable = true; });
  canvas.addEventListener('blur', () => {
    controls.fpsMovement.enable = false;
    controls.fpsMovement.keydown = {};
    controls.fpsMovement.keycode = {};
  });
  canvas.addEventListener('keydown', (event) => {
    if (event.code in controls.fpsMovement.keycodeMoveMapping) event.preventDefault();
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    const axes = { KeyI: [-1, 0, 0], KeyK: [1, 0, 0], KeyJ: [0, -1, 0], KeyL: [0, 1, 0], KeyU: [0, 0, -1], KeyO: [0, 0, 1] };
    if (!(event.code in axes)) return;
    event.preventDefault();
    adjustLevel(new THREE.Vector3(...axes[event.code]).applyQuaternion(camera.quaternion), event.shiftKey ? 5 : 1.5);
  });
  $('#reset-view').addEventListener('click', resetView);
  $('#viewer').append(canvas);
  $('#viewer').classList.add('is-interactive');
  $('#viewer-toolbar').hidden = false;
  $('#controls-hint').hidden = false;
  const resize = () => {
    const { width, height } = $('#viewer').getBoundingClientRect();
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe($('#viewer'));
  resize();
  // Avoid re-quantizing camera-relative centers to float16 during movement.
  const spark = new SparkRenderer({ renderer, accumExtSplats: true });
  scene.add(spark);
  const loader = new GLTFLoader().setMeshoptDecoder(MeshoptDecoder);
  const shaded = new THREE.MeshNormalMaterial({ side: THREE.DoubleSide });
  const colored = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide });
  const pointSize = $('#point-size');
  const points = new THREE.PointsMaterial({ vertexColors: true, size: pointSize.valueAsNumber, sizeAttenuation: true });
  pointSize.addEventListener('input', () => { points.size = pointSize.valueAsNumber; });
  const loaded = new Map();
  let selected = '3dgs';

  function meshStyle() {
    loaded.get('mesh')?.object?.traverse((object) => {
      if (object.isMesh) object.material = $('#mesh-style').value === 'shaded' ? shaded : colored;
    });
  }
  $('#mesh-style').addEventListener('change', meshStyle);

  function updateSelection() {
    const current = loaded.get(selected);
    for (const [name, entry] of loaded) if (entry.object) entry.object.visible = name === selected;
    spark.visible = selected === '3dgs';
    $('#scene-cover').hidden = Boolean(current?.object);
    $('#mesh-style-label').hidden = selected !== 'mesh';
    $('#point-size-label').hidden = selected !== 'pointcloud';
    if (current?.error) {
      viewerMessage(`${modalityLabels[selected]} could not be loaded. Check your connection, then reload to retry, or choose another representation.`);
    } else if (current?.object) {
      viewerMessage('');
    } else {
      const percent = current?.percent;
      viewerMessage(`Loading ${modalityLabels[selected]}${percent === undefined ? '…' : ` · ${percent}%`}${percent === 100 ? ' · Decoding…' : ''}`, true, percent);
    }
  }

  async function loadModality(name, entry) {
    const asset = data.assets[assetNames[name]];
    // Relative asset_base is relative to the site root, including GitHub Pages' /wanderland/ prefix.
    const base = new URL(`${assetBase.replace(/\/$/, '')}/`, document.baseURI);
    const url = new URL(asset.path, base).href;
    const onProgress = (event) => {
      const total = event.total || asset.bytes;
      entry.percent = Math.min(100, Math.round(event.loaded / total * 100));
      if (selected === name) updateSelection();
    };
    try {
      let object;
      if (name === '3dgs') {
        object = new SplatMesh({ url, onProgress });
        await object.initialized;
      } else {
        object = (await loader.loadAsync(url, onProgress)).scene;
        object.traverse((child) => {
          if (child.isPoints) child.material = points;
          if (child.isMesh && !child.geometry.hasAttribute('normal')) child.geometry.computeVertexNormals();
        });
      }
      entry.object = object;
      object.visible = name === selected;
      level.add(object);
      if (name === 'mesh') meshStyle();
    } catch (error) {
      entry.error = error;
      console.error(`Failed to load ${url}`, error);
    }
    updateSelection();
  }

  function selectModality(name) {
    selected = name;
    for (const tab of document.querySelectorAll('[data-modality]')) {
      const active = tab.dataset.modality === name;
      tab.setAttribute('aria-selected', String(active));
      tab.tabIndex = active ? 0 : -1;
    }
    $('#viewer').setAttribute('aria-labelledby', `tab-${name}`);
    if (!loaded.has(name)) {
      const entry = {};
      loaded.set(name, entry);
      loadModality(name, entry);
    }
    updateSelection();
  }
  const tabs = [...document.querySelectorAll('[data-modality]')];
  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => selectModality(tab.dataset.modality));
    tab.addEventListener('keydown', (event) => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
      event.preventDefault();
      const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : (index + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
      tabs[next].focus();
      selectModality(tabs[next].dataset.modality);
    });
  });
  canvas.addEventListener('webglcontextlost', (event) => {
    event.preventDefault();
    renderer.setAnimationLoop(null);
    $('#scene-cover').hidden = false;
    $('#viewer-toolbar').hidden = true;
    viewerMessage('The graphics context was lost. Reload the page to restart the viewer.');
  });
  renderer.setAnimationLoop(() => {
    controls.update(camera);
    renderer.render(scene, camera);
  });
  selectModality('3dgs');
}

async function initScene() {
  try {
    const index = await loadJSON('static/data/scenes.json');
    const id = new URLSearchParams(location.search).get('id');
    // Only fetch IDs present in the site index, never an arbitrary query-string path.
    const entry = index.scenes.find((scene) => scene.id === id);
    if (!entry) throw new Error('Scene not found. Choose a scene from the gallery.');
    const data = await loadJSON(`static/data/scenes/${encodeURIComponent(entry.id)}.json`);
    document.title = `${sceneLabel(entry)} · Wanderland`;
    $('#scene-title').textContent = sceneLabel(entry);
    $('#scene-id').textContent = entry.id;
    $('#scene-cover').src = `static/${data.thumbnails[0].path}`;
    $('#scene-cover').alt = `${sceneLabel(entry)} cover view`;
    showDetails(data);
    $('#scene-content').hidden = false;
    $('#page-status').hidden = true;
    try {
      await initViewer(data, index.asset_base);
    } catch (error) {
      console.error(error);
      $('#scene-cover').hidden = false;
      $('#viewer-toolbar').hidden = true;
      $('#controls-hint').hidden = true;
      viewerMessage('The 3D viewer could not start. Check your connection and WebGL2 support, then reload. Scene details and downloads are available below.');
    }
  } catch (error) {
    console.error(error);
    $('#page-status').textContent = `Unable to open this scene. ${error.message}`;
  }
}

initScene();
