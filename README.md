# BizarroCam

Experimental real-time camera laboratory for Android. GPU-native computer vision,
mesh geometry displacement, face morphing and deliberately bizarre video output.

## Feature set

- **Edge detection** — GPU Sobel field rendered as pulsing neon contours
- **Object detection + tracking** — frame-differencing motion energy map,
  connected-component blob extraction, persistent-ID centroid tracking with
  EMA smoothing, tracked bounding boxes injected into the displacement field
- **Face recognition + tracking** — ML Kit face contours (133 points per face),
  temporal smoothing, per-face bounding boxes, tracking IDs
- **Mesh geometry displacement** — 48x86 vertex grid warped by a radial-basis
  field emitted from face-mesh points + tracked object centers, plus animated
  procedural noise extrusion
- **Face morphing** — inverse RBF pinch/bulge warp driven by live contour points
- **Sliders** — EDGE, DISP, MORPH, GLITCH, ECHO, POSTER, CHROMA, HUE, SLIT,
  KALEIDO, NOISE, SCAN (plus INVERT via randomizer)
- **Camera flip** — front/back lens switching with correct overlay mirroring
- **Recording** — 720x1280 H.264 @ 10 Mbps + AAC audio, rendered off the same
  GL pipeline through a shared-context EGL surface into MediaCodec
- **Snapshot** — PNG frame grabs
- Effects chain: kaleidoscope mirror, scanline glitch displacement, macroblock
  snapping, slit-scan band echo, temporal frame feedback, posterize, hue cycle,
  chromatic aberration, film grain, scanlines, invert

## Architecture

CameraX Preview -> SurfaceTexture (OES) -> pass 1: disp/edge/motion map (FBO)
-> pass 2: displaced-grid composite (stage FBO, ping-pong feedback)
-> pass 3: center-crop blit to screen + vector overlays
-> (optional) same scene re-rendered to encoder EGL surface.

All vision/feature extraction runs on the GPU; the CPU only consumes a
64x36 decimated motion readback for blob tracking.

## Build

GitHub Actions (`build-apk.yml`) builds the debug APK on every push.
See `build-playbooks/Build_APK_via_GitHub_Actions-2026-09-09.md` in the KB.

## License

See LICENSE.
