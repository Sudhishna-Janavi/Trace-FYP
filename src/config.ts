/**
 * Which Trace server this build talks to.
 *
 * The application on the phone is only a set of screens. The transactions,
 * the categories and the explanations all live on a server, and it has to be
 * told which one to ask.
 *
 *   LIVE   the deployed instance. Works on any network, with the laptop shut.
 *          Subject to the free tier's cold start: the first request after a
 *          quiet period can take twenty seconds or more.
 *
 *   LOCAL  a development server on the Mac. Instant, but the phone and the
 *          Mac must be on the same network, and many institutional networks
 *          stop devices talking to each other.
 *
 * Set USE_LOCAL to true while developing. Leave it false for a demonstration,
 * so nothing depends on a laptop being present or a network cooperating.
 */

const USE_LOCAL = false;

/**
 * For USE_LOCAL. `localhost` is right for the web target, which runs on the
 * Mac itself. A real device cannot resolve the Mac's localhost, so testing on
 * the phone against a local server needs this set to the Mac's address on the
 * network, and uvicorn started with `--host 0.0.0.0`, since by default it
 * listens only on the loopback interface and will refuse a connection from
 * anywhere else.
 */
const LOCAL = 'http://localhost:8000';
const LIVE = 'https://trace-fyp.onrender.com';

export const API_BASE = `${USE_LOCAL ? LOCAL : LIVE}/api/v1`;
export const IS_LOCAL = USE_LOCAL;

/**
 * The deployed instance sleeps when idle and takes a while to wake. A request
 * that would normally be refused as timed out is, here, usually just a server
 * starting up, so the timeout is generous and the interface says what is
 * happening rather than failing quietly.
 */
export const REQUEST_TIMEOUT_MS = USE_LOCAL ? 20_000 : 90_000;
export const SLOW_REQUEST_MS = 4_000;
