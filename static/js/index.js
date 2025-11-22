window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE = "./static/interpolation/stacked";
var NUM_INTERP_FRAMES = 240;

var interp_images = [];
function preloadInterpolationImages() {
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + '/' + String(i).padStart(6, '0') + '.jpg';
    interp_images[i] = new Image();
    interp_images[i].src = path;
  }
}

function setInterpolationImage(i) {
  var image = interp_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper').empty().append(image);
}


$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
			slidesToScroll: 1,
			slidesToShow: 3,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/
    preloadInterpolationImages();

    $('#interpolation-slider').on('input', function(event) {
      setInterpolationImage(this.value);
    });
    setInterpolationImage(0);
    $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);

    bulmaSlider.attach();

    setupComparisonVideoSync();

})

function waitForVideoReady(video) {
  return new Promise(resolve => {
    if (video.readyState >= 2) {
      resolve();
      return;
    }
    const onReady = () => {
      video.removeEventListener('loadeddata', onReady);
      video.removeEventListener('canplaythrough', onReady);
      resolve();
    };
    video.addEventListener('loadeddata', onReady);
    video.addEventListener('canplaythrough', onReady);
  });
}

function syncVideoGroup(videos) {
  if (!videos.length) {
    return;
  }
  const master = videos[0];
  const playGroup = () => {
    videos.forEach(video => {
      if (video.paused) {
        video.play().catch(() => {});
      }
    });
  };
  const alignTimes = () => {
    const referenceTime = master.currentTime;
    videos.forEach((video, index) => {
      if (index === 0) {
        return;
      }
      const delta = Math.abs(video.currentTime - referenceTime);
      if (delta > 0.05) {
        video.currentTime = referenceTime;
      }
    });
  };

  videos.forEach(video => {
    video.loop = true;
    video.muted = true;
    video.autoplay = false;
    video.playsInline = true;
    video.pause();
    try {
      video.currentTime = 0;
    } catch (error) {
      // Some browsers may block currentTime if metadata isn't loaded yet.
    }
  });

  Promise.all(videos.map(waitForVideoReady)).then(() => {
    videos.forEach(video => {
      try {
        video.currentTime = 0;
      } catch (error) {
        // Ignore seek errors before metadata is ready.
      }
    });
    playGroup();
  });

  master.addEventListener('timeupdate', alignTimes);
  master.addEventListener('seeked', alignTimes);
  master.addEventListener('play', playGroup);
  master.addEventListener('pause', () => {
    videos.forEach(video => {
      if (!video.paused) {
        video.pause();
      }
    });
  });
  master.addEventListener('ended', () => {
    videos.forEach(video => {
      try {
        video.currentTime = 0;
      } catch (error) {
        // Ignore seek errors for safety.
      }
    });
    playGroup();
  });
}

function setupComparisonVideoSync() {
  const sliders = document.querySelectorAll('img-comparison-slider');
  sliders.forEach(slider => {
    if (slider.dataset.synced === 'true') {
      return;
    }
    const videos = slider.querySelectorAll('video');
    if (videos.length >= 2) {
      slider.dataset.synced = 'true';
      syncVideoGroup(Array.from(videos));
    }
  });
}
