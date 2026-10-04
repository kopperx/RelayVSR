(() => {
'use strict';
const D = window.RELAY;
const $ = (s, root = document) => root.querySelector(s);
const $$ = (s, root = document) => [...root.querySelectorAll(s)];
const clipById = id => D.clips.find(c => c.id === id);
const methodName = id => D.methods.find(m => m.id === id)?.label || id;
const players = new Set();
const timeLabel = time => `${Math.floor(time / 60)}:${String(Math.floor(time % 60)).padStart(2, '0')}`;
const escapeHTML = text => String(text).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
class MediaPlayer {
  constructor(root, clip, options = {}) {
    this.root = root; this.clip = clip; this.options = options;
    this.method = options.method || 'input'; this.compare = true;
    this.layout = options.layout || 'wipe'; this.running = false; this.loaded = false; this.revision = 0;
    root.innerHTML = `<div class="media-player" tabindex="0" aria-label="${escapeHTML(clip.title)} video player"><div class="video-stage"><div class="video-layer output-layer"><video muted playsinline preload="none" poster="${clip.poster}" aria-label="RelayVSR output"></video></div><div class="video-layer reference-layer"><video muted playsinline preload="none" ${this.method === 'input' ? `poster="${clip.inputPoster}"` : ''} aria-label="${escapeHTML(methodName(this.method))}"></video></div><span class="video-label">RelayVSR (Output)</span><span class="video-label reference-label">${escapeHTML(methodName(this.method))}</span><div class="wipe-control"><span class="wipe-handle" aria-hidden="true">↔</span><input type="range" min="0" max="100" value="50" aria-label="Comparison divider"></div><button class="center-play" aria-label="Play ${escapeHTML(clip.title)}">▶</button></div><div class="player-controls"><button class="play-toggle" aria-label="Play">▶</button><span class="player-time">0:00 / ${timeLabel(clip.frames / clip.fps)}</span><input class="seek" type="range" min="0" max="${clip.frames - 1}" value="0" step="1" aria-label="Video frame"><select aria-label="Playback speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select><button class="fullscreen" aria-label="Fullscreen">⛶</button></div><div class="player-status" role="status"></div></div>`;
    this.el = $('.media-player',root); this.ours = $('.output-layer video',root); this.reference = $('.reference-layer video',root);
    this.seek = $('.seek',root); this.status = $('.player-status',root);
    if(options.onPrevious && options.onNext) {
      const stage = $('.video-stage',root);
      stage.insertAdjacentHTML('beforeend','<button class="scene-arrow scene-previous" aria-label="Previous video">‹</button><button class="scene-arrow scene-next" aria-label="Next video">›</button>');
      $('.scene-previous',root).addEventListener('click',options.onPrevious);
      $('.scene-next',root).addEventListener('click',options.onNext);
    }
    this.ours.muted = this.reference.muted = true;
    this.applyLayout();
    $('.play-toggle',root).addEventListener('click', () => this.toggle());
    $('.center-play',root).addEventListener('click', () => this.toggle());
    $('.wipe-control input',root).addEventListener('input', e => this.el.style.setProperty('--split',`${e.target.value}%`));
    this.seek.addEventListener('input', () => { this.pause(); this.seekFrame(Number(this.seek.value) + 1); });
    $('select',root).addEventListener('change', e => { this.ours.playbackRate = this.reference.playbackRate = Number(e.target.value); });
    $('.fullscreen',root).addEventListener('click', async () => {
      try { if(document.fullscreenElement) await document.exitFullscreen(); else if(this.el.requestFullscreen) await this.el.requestFullscreen(); else this.ours.webkitEnterFullscreen?.(); }
      catch { this.status.textContent = 'Fullscreen is unavailable in this browser.'; }
    });
    this.el.addEventListener('keydown', e => {
      if(/INPUT|SELECT|BUTTON|A/.test(e.target.tagName)) return;
      if(e.code === 'Space') { e.preventDefault(); this.toggle(); }
      if(e.code === 'ArrowRight' || e.code === 'ArrowLeft') { e.preventDefault(); this.pause(); this.seekFrame(Math.round(this.ours.currentTime * clip.fps) + (e.code === 'ArrowRight' ? 2 : 0)); }
    });
    this.ours.addEventListener('timeupdate', () => this.update());
    this.ours.addEventListener('ended', () => { if(players.has(this) && this.running) this.play(); });
    for(const video of [this.ours,this.reference]) video.addEventListener('error', () => { if(!players.has(this)) return; this.pause(); this.status.textContent = 'Video could not load. Check that the local media folder is available.'; });
    players.add(this);
    // Comparison baselines have no fabricated poster: decode their own first frame.
    if(options.compare && !options.autoplay && this.method !== 'input') this.seekTo(0);
    if(options.autoplay) {
      this.observer = new IntersectionObserver(entries => {
        const entry = entries[0];
        if(!entry.isIntersecting || entry.intersectionRatio < 0.45) { this.pause(); return; }
        if(!document.hidden && !document.querySelector('dialog[open]')) this.play();
      }, {threshold:0.45});
      this.observer.observe(this.el);
    }
  }
  applyLayout() {
    this.el.classList.toggle('is-comparing',this.compare);
    this.el.classList.toggle('side-layout',this.compare && this.layout === 'side');
  }
  load(video, method) {
    if(video.getAttribute('src')) return;
    video.src = this.clip.sources[method]; video.preload = 'auto'; video.load();
  }
  ready(video, method) {
    this.load(video,method);
    if(video.readyState >= 1) return Promise.resolve();
    return new Promise((resolve,reject) => {
      const finish = fn => { clearTimeout(timer); video.removeEventListener('loadedmetadata',ok);video.removeEventListener('error',bad);fn(); };
      const ok = () => finish(resolve), bad = () => finish(() => reject(new Error('Media unavailable')));
      const timer = setTimeout(bad,90000);
      video.addEventListener('loadedmetadata',ok,{once:true});video.addEventListener('error',bad,{once:true});
    });
  }
  async seekTo(time) {
    const version = ++this.revision;
    this.status.textContent = 'Loading video…';
    try {
      await Promise.all([this.ready(this.ours,'relayvsr'), ...(this.compare ? [this.ready(this.reference,this.method)] : [])]);
      if(version !== this.revision || !players.has(this)) return;
      const end = (this.clip.frames - 1) / this.clip.fps;
      const t = Math.max(0,Math.min(time,end,this.ours.duration || end));
      this.ours.currentTime = t;
      if(this.compare) this.reference.currentTime = Math.min(t,Math.max(0,this.reference.duration - 1 / this.clip.fps));
      this.status.textContent = ''; this.update();
    } catch { if(version === this.revision && players.has(this)) this.status.textContent = 'Unable to load this video. Please try again.'; }
  }
  seekFrame(frame) { return this.seekTo((Math.max(1,Math.min(this.clip.frames,frame)) - 1) / this.clip.fps); }
  async play() {
    for(const player of players) if(player !== this) player.pause();
    this.running = true; const version = ++this.revision;
    this.status.textContent = 'Loading video…'; this.drawState();
    try {
      await Promise.all([this.ready(this.ours,'relayvsr'),...(this.compare ? [this.ready(this.reference,this.method)] : [])]);
      if(version !== this.revision || !this.running) return;
      if(this.ours.ended || this.ours.currentTime >= (this.clip.frames - 1) / this.clip.fps) this.ours.currentTime = 0;
      if(this.compare) { this.reference.currentTime = this.ours.currentTime; this.reference.playbackRate = this.ours.playbackRate; }
      await Promise.all([this.ours.play(),...(this.compare ? [this.reference.play()] : [])]);
      if(version !== this.revision || !this.running) { this.ours.pause();this.reference.pause();return; }
      this.status.textContent = ''; this.tick();
    } catch { if(version === this.revision) { this.pause();this.status.textContent = 'Playback did not start. Press play to retry.'; } }
  }
  tick() {
    cancelAnimationFrame(this.raf);
    if(!this.running) return;
    if(this.compare && Math.abs(this.reference.currentTime - this.ours.currentTime) > .09 && !this.reference.seeking) this.reference.currentTime = this.ours.currentTime;
    this.update(); this.raf = requestAnimationFrame(() => this.tick());
  }
  pause() { ++this.revision; this.status.textContent = ''; this.running = false;this.ours.pause();this.reference.pause();cancelAnimationFrame(this.raf);this.drawState(); }
  toggle() { this.running ? this.pause() : this.play(); }
  drawState() { this.el.classList.toggle('is-playing',this.running);const button = $('.play-toggle',this.root);button.textContent = this.running ? 'Ⅱ' : '▶';button.setAttribute('aria-label',this.running ? 'Pause' : 'Play'); }
  update() {
    if(!players.has(this)) return;
    const frame = Math.min(this.clip.frames - 1,Math.round(this.ours.currentTime * this.clip.fps));
    this.seek.value = frame;this.seek.setAttribute('aria-valuetext',`Frame ${frame+1} of ${this.clip.frames}`);
    $('.player-time',this.root).textContent = `${timeLabel(this.ours.currentTime)} / ${timeLabel(this.clip.frames / this.clip.fps)}`;
  }
  destroy() { this.observer?.disconnect(); this.pause();players.delete(this);for(const video of [this.ours,this.reference]) { video.removeAttribute('src');video.load(); } this.root.replaceChildren(); }
}
function pill(clip,selected) { return `<button class="scene-pill" data-id="${clip.id}" aria-pressed="${selected}"><img src="${clip.poster}" alt="" loading="lazy"><span>${escapeHTML(clip.title)}</span></button>`; }
function home() {
  let hero, long, category = 'synthetic';
  const categories = ['synthetic','real','aigc'];
  const groups = Object.fromEntries(categories.map(key => [key,D.clips.filter(c => c.category === key)]));
  const remembered = Object.fromEntries(categories.map(key => [key,groups[key][0].id]));
  let current = groups[category][0];
  function stepHero(direction) {
    const clips = groups[category];
    const index = clips.findIndex(c => c.id === current.id);
    const restoreFocus = document.activeElement?.classList.contains('scene-arrow');
    setHero(clips[(index + direction + clips.length) % clips.length].id);
    if(restoreFocus) $(direction < 0 ? '.scene-previous' : '.scene-next',hero.el).focus({preventScroll:true});
  }
  function setHero(id) {
    current = clipById(id);category = current.category;remembered[category] = id;
    hero?.destroy();
    hero = new MediaPlayer($('#hero-player'),current,{autoplay:true,onPrevious:() => stepHero(-1),onNext:() => stepHero(1)});
    $('#hero-name').textContent = current.title;
    const clips = groups[category];
    $('#hero-position').textContent = (clips.indexOf(current)+1) + ' / ' + clips.length;
    $$('.result-categories [data-category]').forEach(button => button.setAttribute('aria-pressed',String(button.dataset.category === category)));
    $('#hero-selector').innerHTML = clips.map(c => '<button class="sample-button" data-id="'+c.id+'" aria-pressed="'+(c.id===id)+'"><img src="'+c.poster+'" alt="" loading="lazy"><span>'+(c.group==='VideoLQ'?'VideoLQ · ':'Sample ')+c.sample+'</span></button>').join('');
  }
  $('.result-categories').addEventListener('click',event => {
    const button = event.target.closest('[data-category]');
    if(button) setHero(remembered[button.dataset.category]);
  });
  $('#hero-selector').addEventListener('click',event => {
    const button = event.target.closest('[data-id]');if(button) setHero(button.dataset.id);
  });
  setHero(current.id);
  function setLong(id) {
    long?.destroy();long = new MediaPlayer($('#long-player'),clipById(id),{autoplay:true});
    $('#long-selector').innerHTML = D.long.map(key => '<button data-id="'+key+'" aria-pressed="'+(key===id)+'">'+escapeHTML(clipById(key).title)+'</button>').join('');
  }
  $('#long-selector').addEventListener('click',event => {if(event.target.dataset.id) setLong(event.target.dataset.id);});
  setLong(D.long[1] || D.long[0]);
}
function comparisons() {
  const collections = [
    {id:'synthetic',label:'Synthetic benchmarks',match:c=>c.category==='synthetic'},
    {id:'videolq',label:'Real-world · VideoLQ',match:c=>c.category==='real'&&c.group==='VideoLQ'},
    {id:'real',label:'Real-world · Additional',match:c=>c.category==='real'&&c.group!=='VideoLQ'},
    {id:'aigc',label:'AIGC videos',match:c=>c.category==='aigc'},
    {id:'streaming',label:'Long-form · 1,000 frames',match:c=>c.category==='streaming'}
  ];
  const params = new URLSearchParams(location.search);
  let clip = clipById(params.get('scene')) || D.clips.find(c=>c.category===params.get('category')) || D.clips[0];
  let collection = collections.find(c=>c.match(clip));
  let method = 'flashvsr-tiny', player, available;
  const groupSelect=$('#collection-select'),sceneSelect=$('#scene-select'),methodSelect=$('#method-select');
  groupSelect.innerHTML = collections.map(c=>`<option value="${c.id}">${c.label}</option>`).join('');
  function render() {
    groupSelect.value=collection.id;available=D.clips.filter(collection.match);
    sceneSelect.innerHTML=available.map(c=>`<option value="${c.id}">${escapeHTML(c.title)}</option>`).join('');sceneSelect.value=clip.id;
    if(!clip.sources[method])method='input';
    methodSelect.innerHTML=D.methods.filter(m=>m.id!=='relayvsr'&&clip.sources[m.id]).map(m=>`<option value="${m.id}">${m.label}</option>`).join('');methodSelect.value=method;
    player?.destroy();player=new MediaPlayer($('#comparison-viewer'),clip,{compare:true,method,layout:'wipe',autoplay:true});
    $('#comparison-meta').textContent=`Sequence ${clip.sample} · ${clip.resolution} · ${clip.frames.toLocaleString()} frames · ${clip.fps.toFixed(2).replace('.00','')} fps`;
    $('#scene-strip').innerHTML=available.map(c=>pill(c,c.id===clip.id)).join('');
    const index=available.indexOf(clip);$('#previous-scene').disabled=index===0;$('#next-scene').disabled=index===available.length-1;

  }
  groupSelect.addEventListener('change',()=>{collection=collections.find(c=>c.id===groupSelect.value);clip=D.clips.find(collection.match);render();});
  sceneSelect.addEventListener('change',()=>{clip=clipById(sceneSelect.value);render();});
  methodSelect.addEventListener('change',()=>{method=methodSelect.value;render();});
  $('#scene-strip').addEventListener('click',e=>{const b=e.target.closest('[data-id]');if(b){clip=clipById(b.dataset.id);render();}});
  $('#previous-scene').addEventListener('click',()=>{clip=available[available.indexOf(clip)-1];render();});
  $('#next-scene').addEventListener('click',()=>{clip=available[available.indexOf(clip)+1];render();});
  render();
}
document.addEventListener('visibilitychange',()=>{if(document.hidden)for(const player of players)player.pause();});
window.addEventListener('pagehide',()=>{for(const player of players)player.pause();});
home();
comparisons();
})();
