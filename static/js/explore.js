export const number = new Intl.NumberFormat('en-US', { maximumFractionDigits: 1 });
const compact = new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 });
export const sceneLabel = (scene) => `Scene ${String(scene.number).padStart(2, '0')}`;

export function formatBytes(bytes) {
  if (bytes < 1000) return `${bytes} B`;
  const units = ['kB', 'MB', 'GB', 'TB'];
  const power = Math.min(Math.floor(Math.log10(bytes) / 3), units.length);
  return `${number.format(bytes / 1000 ** power)} ${units[power - 1]}`;
}

export async function loadJSON(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return response.json();
}

function sceneCard(scene) {
  const card = document.createElement('a');
  card.className = 'scene-card visual-highlight-card';
  card.href = `scene.html?id=${encodeURIComponent(scene.id)}`;
  card.innerHTML = `<img loading="lazy" decoding="async" width="640" height="360">
    <div class="visual-highlight-caption">
      <div class="card-heading"><h3></h3><span></span></div>
      <dl class="card-stats"></dl>
    </div>`;
  const image = card.querySelector('img');
  image.src = `static/${scene.thumbnails[0]}`;
  image.alt = `${sceneLabel(scene)} preview`;
  card.querySelector('h3').textContent = sceneLabel(scene);
  card.querySelector('.card-heading span').textContent = `${scene.id.slice(0, 8)}…`;
  const stats = [
    ['Fisheye images', compact.format(scene.stats.fisheye_images)],
    ['LiDAR points', compact.format(scene.stats.las_points)],
    ['Gaussians', compact.format(scene.stats.gaussians)],
    ['Ground coverage', `${number.format(scene.stats.ground_coverage_m2)} m²`],
    ['Trajectory', `${number.format(scene.stats.trajectory_length_m)} m`],
  ];
  for (const [label, value] of stats) {
    const item = document.createElement('div');
    const term = document.createElement('dt');
    const detail = document.createElement('dd');
    term.textContent = label;
    detail.textContent = value;
    item.append(term, detail);
    card.querySelector('dl').append(item);
  }
  let timer;
  let frame = 0;
  function stopPreview() {
    clearInterval(timer);
    timer = undefined;
    frame = 0;
    image.src = `static/${scene.thumbnails[0]}`;
  }
  card.addEventListener('pointerenter', (event) => {
    if (event.pointerType === 'touch' || matchMedia('(prefers-reduced-motion: reduce)').matches || scene.thumbnails.length < 2) return;
    for (const path of scene.thumbnails.slice(1)) new Image().src = `static/${path}`;
    timer = setInterval(() => {
      frame = (frame + 1) % scene.thumbnails.length;
      image.src = `static/${scene.thumbnails[frame]}`;
    }, 1000);
  });
  card.addEventListener('pointerleave', stopPreview);
  document.addEventListener('visibilitychange', () => { if (document.hidden) stopPreview(); });
  return card;
}

async function initGallery() {
  const status = document.querySelector('#gallery-status');
  try {
    const { scenes } = await loadJSON('static/data/scenes.json');
    document.querySelector('#scene-grid').replaceChildren(...scenes.map(sceneCard));
    status.textContent = `${scenes.length} scenes available · Open a scene to explore in 3D`;
  } catch (error) {
    console.error(error);
    status.textContent = 'The scene gallery could not be loaded. Please reload the page.';
  }
  const video = document.querySelector('.explorer-banner video');
  const pause = document.querySelector('#banner-pause');
  const updateButton = () => {
    pause.textContent = video.paused ? 'Play video' : 'Pause video';
    pause.setAttribute('aria-label', `${video.paused ? 'Play' : 'Pause'} banner video`);
  };
  if (matchMedia('(prefers-reduced-motion: reduce)').matches) video.pause();
  video.addEventListener('play', updateButton);
  video.addEventListener('pause', updateButton);
  pause.addEventListener('click', async () => {
    if (video.paused) {
      try { await video.play(); } catch (error) { console.error('Banner playback failed:', error); }
    } else video.pause();
    updateButton();
  });
  updateButton();
}

if (document.querySelector('#scene-grid')) initGallery();
