//! Camera state: an orbit/fly controller plus the matrices the renderer reads each frame.
//!
//! The camera is intentionally plain data with public fields — a host application (a flight sim,
//! an editor) is expected to drive `position`, `yaw`, `pitch` or `orientation` directly and call
//! [`Camera::update`] only when it wants the built-in keyboard controller. Nothing here talks to
//! the GPU; [`Camera::view`] and [`Camera::proj`] are consumed by
//! [`crate::renderer::Renderer::render`].

use glam::{Mat4, Quat, Vec3};
use std::collections::HashSet;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

/// How the built-in controller interprets motion.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CameraMode {
    /// The camera looks at `target` from `distance` away; moving translates the target, and the
    /// scroll wheel changes the distance. Best for inspecting a model.
    Orbit,
    /// The camera sits at `position` and flies freely; the scroll wheel changes `speed`. Best for
    /// walking or flying through a world.
    Fly,
}

/// A perspective camera with an orbit/fly controller.
///
/// Orientation is `yaw`/`pitch` in radians (yaw around +Y, pitch positive looking up) unless
/// `orientation` is set, in which case a full quaternion wins and roll is possible. The
/// projection is reverse-Z with an infinite far plane, so `z_far` is only used for culling and
/// shadow-range decisions, never for the depth mapping.
pub struct Camera {
    /// Which controller mode the built-in event handling uses.
    pub mode: CameraMode,
    /// Look-at point in orbit mode; also the pivot that mouse-drag rotation circles.
    pub target: Vec3,
    /// Eye-to-target distance in orbit mode, and the pivot distance when switching to fly.
    pub distance: f32,
    /// Eye position in fly mode. Ignored in orbit mode, where the eye is derived from `target`.
    pub position: Vec3,
    /// Heading in radians: 0 looks down -Z, positive rotates toward +X.
    pub yaw: f32,
    /// Elevation in radians, clamped to just under ±90° so the basis never degenerates.
    pub pitch: f32,
    /// Vertical field of view in radians.
    pub fov_y: f32,
    /// Near plane distance; with reverse-Z this dominates depth precision.
    pub z_near: f32,
    /// Far plane used for culling and shadow range. The projection itself is infinite.
    pub z_far: f32,
    /// Fly-mode movement speed in metres per second.
    pub speed: f32,
    /// Radians of rotation per pixel of mouse motion.
    pub sensitivity: f32,
    /// Exposure offset in stops, driven by `[` / `]`. The host forwards it to
    /// [`crate::renderer::Settings::exposure_ev`].
    pub exposure_ev: f32,
    /// Full 3-axis orientation (camera looks down local -Z, local +Y is up).
    /// When set it overrides yaw/pitch, allowing roll (cockpit / chase cameras).
    pub orientation: Option<Quat>,
    keys: HashSet<KeyCode>,
    rotating: bool,
    panning: bool,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            mode: CameraMode::Orbit,
            target: Vec3::new(0.0, 1.0, 0.0),
            distance: 9.0,
            position: Vec3::new(0.0, 2.0, 8.0),
            yaw: -0.6,
            pitch: -0.15,
            fov_y: 45.0_f32.to_radians(),
            z_near: 0.05,
            z_far: 500.0,
            speed: 6.0,
            sensitivity: 0.0025,
            exposure_ev: 0.0,
            orientation: None,
            keys: HashSet::new(),
            rotating: false,
            panning: false,
        }
    }
}

impl Camera {
    /// Unit vector the camera looks along.
    ///
    /// Derived from `orientation` when one is set, otherwise from `yaw`/`pitch`.
    pub fn forward(&self) -> Vec3 {
        if let Some(q) = self.orientation {
            return (q * Vec3::NEG_Z).normalize();
        }
        let cp = self.pitch.cos();
        Vec3::new(self.yaw.sin() * cp, self.pitch.sin(), -self.yaw.cos() * cp).normalize()
    }

    /// Unit vector pointing to the camera's right (its local +X).
    ///
    /// In yaw/pitch mode this is the horizontal right vector — `forward × Y`, so it stays level
    /// with the ground even when the camera looks up or down.
    pub fn right(&self) -> Vec3 {
        if let Some(q) = self.orientation {
            return (q * Vec3::X).normalize();
        }
        self.forward().cross(Vec3::Y).normalize_or_zero()
    }

    /// Unit vector pointing out of the top of the camera (its local +Y).
    ///
    /// In yaw/pitch mode this is rebuilt from `right` and `forward`, which is why roll is
    /// impossible without setting `orientation`.
    pub fn up(&self) -> Vec3 {
        if let Some(q) = self.orientation {
            return (q * Vec3::Y).normalize();
        }
        self.right().cross(self.forward()).normalize_or_zero()
    }

    /// World-space eye position the view matrix is built from.
    ///
    /// Orbit mode derives the eye by walking `distance` backwards from `target` along `forward`,
    /// so the two stay consistent when either is changed by the host.
    pub fn eye(&self) -> Vec3 {
        match self.mode {
            CameraMode::Orbit => self.target - self.forward() * self.distance,
            CameraMode::Fly => self.position,
        }
    }

    /// Right-handed world-to-view matrix (camera looks down its local -Z).
    ///
    /// The up vector is world +Y in yaw/pitch mode — deliberately, rather than the camera's own
    /// `up()` — so dragging the mouse to the pitch clamp cannot flip the view; with a full
    /// orientation the camera's up is used and roll takes effect.
    #[allow(deprecated)]
    pub fn view(&self) -> Mat4 {
        let eye = self.eye();
        let up = if self.orientation.is_some() {
            self.up()
        } else {
            Vec3::Y
        };
        Mat4::look_to_rh(eye, self.forward(), up)
    }

    /// Reverse-Z perspective projection with an infinite far plane, for the given aspect ratio.
    ///
    /// Mapping depth this way puts almost all of the float's resolution in the first few metres:
    /// with `Depth32Float` the near plane stays sub-millimetre-accurate out to tens of kilometres,
    /// which is what lets the same projection serve a 2 m-scale scene and a 100 km flight view.
    /// `aspect` is clamped so a zero-height surface produces a finite matrix instead of NaNs.
    #[allow(deprecated)]
    pub fn proj(&self, aspect: f32) -> Mat4 {
        // Reverse-Z with an infinite far plane: depth 1 at the near plane, 0 at infinity.
        // Gives near-uniform precision out to tens of kilometres with Depth32Float.
        Mat4::perspective_infinite_reverse_rh(self.fov_y, aspect.max(0.0001), self.z_near)
    }

    /// Feed a window event to the controller: Tab, WASD/QE, mouse buttons, wheel, focus loss.
    ///
    /// Handles only the keys the demo viewer uses (orbit/fly toggle, movement keys, exposure
    /// brackets). Everything else is ignored, so the host can route the same events elsewhere
    /// without conflict.
    pub fn process_event(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    match event.state {
                        ElementState::Pressed => {
                            if code == KeyCode::Tab {
                                self.mode = match self.mode {
                                    CameraMode::Orbit => {
                                        self.position = self.eye();
                                        CameraMode::Fly
                                    }
                                    CameraMode::Fly => {
                                        self.target =
                                            self.position + self.forward() * self.distance;
                                        CameraMode::Orbit
                                    }
                                };
                            } else {
                                self.keys.insert(code);
                            }
                        }
                        ElementState::Released => {
                            self.keys.remove(&code);
                        }
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => match button {
                MouseButton::Left | MouseButton::Right => {
                    self.rotating = *state == ElementState::Pressed;
                }
                MouseButton::Middle => {
                    self.panning = *state == ElementState::Pressed;
                }
                _ => {}
            },
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.1,
                };
                match self.mode {
                    CameraMode::Orbit => {
                        self.distance = (self.distance * (1.0 - scroll * 0.1)).clamp(0.5, 300.0);
                    }
                    CameraMode::Fly => {
                        self.speed = (self.speed * (1.0 + scroll * 0.1)).clamp(0.1, 200.0);
                    }
                }
            }
            WindowEvent::Focused(false) => {
                // Without this, alt-tabbing away with W held leaves the camera drifting and the
                // key stuck until it is pressed again.
                self.keys.clear();
                self.rotating = false;
                self.panning = false;
            }
            _ => {}
        }
    }

    /// Apply relative mouse motion (raw device deltas, not cursor positions).
    ///
    /// Rotation is clamped just under ±90° of pitch so `right()` never becomes the zero vector;
    /// panning scales with `distance` so dragging feels the same whether you are close to the
    /// target or far from it.
    pub fn process_mouse_motion(&mut self, dx: f32, dy: f32) {
        if self.rotating {
            self.yaw += dx * self.sensitivity;
            self.pitch = (self.pitch - dy * self.sensitivity).clamp(-1.5533, 1.5533);
        }
        if self.panning {
            let right = self.right();
            let up = self.up();
            let pan = (-right * dx + up * dy) * 0.002 * self.distance.max(1.0);
            match self.mode {
                CameraMode::Orbit => self.target += pan,
                CameraMode::Fly => self.position += pan,
            }
        }
    }

    /// Advance held-key movement and exposure by `dt` seconds.
    ///
    /// Movement is normalised so diagonals are not faster, Shift triples the speed, and in orbit
    /// mode the target moves instead of the eye (so the orbit stays centred). Call once per frame
    /// before rendering; a host that sets the camera itself can skip it entirely.
    pub fn update(&mut self, dt: f32) {
        let mut dir = Vec3::ZERO;
        if self.keys.contains(&KeyCode::KeyW) {
            dir += self.forward();
        }
        if self.keys.contains(&KeyCode::KeyS) {
            dir -= self.forward();
        }
        if self.keys.contains(&KeyCode::KeyD) {
            dir += self.right();
        }
        if self.keys.contains(&KeyCode::KeyA) {
            dir -= self.right();
        }
        if self.keys.contains(&KeyCode::KeyE) || self.keys.contains(&KeyCode::Space) {
            dir += Vec3::Y;
        }
        if self.keys.contains(&KeyCode::KeyQ) || self.keys.contains(&KeyCode::ControlLeft) {
            dir -= Vec3::Y;
        }
        if dir.length_squared() > 0.0 {
            let boost = if self.keys.contains(&KeyCode::ShiftLeft) {
                3.0
            } else {
                1.0
            };
            let motion = dir.normalize() * self.speed * boost * dt;
            match self.mode {
                CameraMode::Orbit => self.target += motion,
                CameraMode::Fly => self.position += motion,
            }
        }
        if self.keys.contains(&KeyCode::BracketLeft) {
            self.exposure_ev -= dt * 0.8;
        }
        if self.keys.contains(&KeyCode::BracketRight) {
            self.exposure_ev += dt * 0.8;
        }
        self.exposure_ev = self.exposure_ev.clamp(-8.0, 8.0);
    }
}
