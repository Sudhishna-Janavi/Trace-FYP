/* Trace service worker.
 *
 * What it is for: the application launches from a home-screen icon, and a
 * launch that waits on the network before painting anything does not feel
 * like an application. The shell is therefore cached and served immediately.
 *
 * What it deliberately does not do: serve stale financial data. A cached
 * balance is worse than a slow one, so the strategy differs by request:
 *
 *   /api/ and anything not GET   network only, never cached
 *   /static/                     cache first, refreshed in the background
 *   page navigations             network first, cache only as a fallback
 *
 * The version string is what retires an old cache. Bump it on release; the
 * activate handler then deletes every cache that is not the current one.
 */
const VERSION = 'trace-v1';
const SHELL = [
  '/static/trace.css',
  '/static/icons/icon-192.png',
  '/static/icons/icon-512.png',
  '/static/icons/apple-touch-icon.png',
  '/offline',
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(VERSION)
      // one failure must not abort the whole install, so each is attempted
      // on its own and a miss is tolerated
      .then(cache => Promise.allSettled(SHELL.map(url => cache.add(url))))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys()
      .then(keys => Promise.all(
        keys.filter(key => key !== VERSION).map(key => caches.delete(key))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', event => {
  const request = event.request;
  const url = new URL(request.url);

  // Only this origin, only GET. Everything else — sign-in, corrections,
  // confirmations — goes straight to the network untouched.
  if (request.method !== 'GET' || url.origin !== self.location.origin) return;

  // Explanations and review actions must be live.
  if (url.pathname.startsWith('/api/')) return;

  // Static assets change only on release, so answer from cache and refresh
  // in the background.
  if (url.pathname.startsWith('/static/')) {
    event.respondWith(
      caches.match(request).then(hit => {
        const network = fetch(request).then(response => {
          if (response && response.ok) {
            const copy = response.clone();
            caches.open(VERSION).then(cache => cache.put(request, copy));
          }
          return response;
        }).catch(() => hit);
        return hit || network;
      })
    );
    return;
  }

  // Pages: the network decides, and the cache is only a fallback. A signed-out
  // redirect or a fresh balance is always preferred to whatever was last seen.
  if (request.mode === 'navigate') {
    event.respondWith(
      fetch(request)
        .then(response => {
          if (response && response.ok) {
            const copy = response.clone();
            caches.open(VERSION).then(cache => cache.put(request, copy));
          }
          return response;
        })
        .catch(() => caches.match(request)
          .then(hit => hit || caches.match('/offline')))
    );
  }
});
