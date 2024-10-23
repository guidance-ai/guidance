var cache_require = window.require;

window.addEventListener('load', function() {
  window.require = cache_require;
});
