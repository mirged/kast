use glam::{Mat4, Quat, Vec3};
use std::collections::HashSet;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CameraMode {
    Orbit,
    Fly,
}

pub struct Camera {
    pub mode: CameraMode,
    pub target: Vec3,
    pub distance: f32,
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub fov_y: f32,
    pub z_near: f32,
    pub z_far: f32,
    pub speed: f32,
    pub sensitivity: f32,
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
    pub fn forward(&self) -> Vec3 {
        if let Some(q) = self.orientation {
            return (q * Vec3::NEG_Z).normalize();
        }
        let cp = self.pitch.cos();
        Vec3::new(self.yaw.sin() * cp, self.pitch.sin(), -self.yaw.cos() * cp).normalize()
    }

    pub fn right(&self) -> Vec3 {
        if let Some(q) = self.orientation {
            return (q * Vec3::X).normalize();
        }
        self.forward().cross(Vec3::Y).normalize_or_zero()
    }

    pub fn up(&self) -> Vec3 {
        if let Some(q) = self.orientation {
            return (q * Vec3::Y).normalize();
        }
        self.right().cross(self.forward()).normalize_or_zero()
    }

    pub fn eye(&self) -> Vec3 {
        match self.mode {
            CameraMode::Orbit => self.target - self.forward() * self.distance,
            CameraMode::Fly => self.position,
        }
    }

    #[allow(deprecated)]
    pub fn view(&self) -> Mat4 {
        let eye = self.eye();
        let up = if self.orientation.is_some() { self.up() } else { Vec3::Y };
        Mat4::look_to_rh(eye, self.forward(), up)
    }

    #[allow(deprecated)]
    pub fn proj(&self, aspect: f32) -> Mat4 {
        // Reverse-Z with an infinite far plane: depth 1 at the near plane, 0 at infinity.
        // Gives near-uniform precision out to tens of kilometres with Depth32Float.
        Mat4::perspective_infinite_reverse_rh(self.fov_y, aspect.max(0.0001), self.z_near)
    }

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
                self.keys.clear();
                self.rotating = false;
                self.panning = false;
            }
            _ => {}
        }
    }

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
