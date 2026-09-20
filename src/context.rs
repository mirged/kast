use anyhow::{Context, Result, anyhow};
use std::sync::Arc;
use winit::dpi::PhysicalSize;
use winit::window::Window;

use crate::gpu;

pub struct GpuContext {
    #[allow(dead_code)]
    pub instance: wgpu::Instance,
    #[allow(dead_code)]
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: Option<wgpu::Surface<'static>>,
    pub config: Option<wgpu::SurfaceConfiguration>,
    pub size: PhysicalSize<u32>,
    pub target_format: wgpu::TextureFormat,
    pub srgb_target: bool,
    #[allow(dead_code)]
    pub adapter_info: wgpu::AdapterInfo,
}

impl GpuContext {
    pub async fn new_windowed(window: Arc<Window>) -> Result<Self> {
        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(window.clone())
            .context("failed to create wgpu surface")?;

        let adapter = request_adapter(&instance, Some(&surface)).await?;
        let (device, queue) = create_device(&adapter).await?;

        let caps = surface.get_capabilities(&adapter);
        let format = choose_format(&caps.formats);
        let srgb_target = format.is_srgb();
        let adapter_info = adapter.get_info();

        let size = window.inner_size();
        let size = PhysicalSize::new(size.width.max(1), size.height.max(1));

        let mut config = surface
            .get_default_config(&adapter, size.width, size.height)
            .ok_or_else(|| anyhow!("surface is not compatible with the selected adapter"))?;
        config.format = format;
        config.present_mode = choose_present_mode(&caps.present_modes);
        config.usage = wgpu::TextureUsages::RENDER_ATTACHMENT;
        surface.configure(&device, &config);

        log::info!(
            "GPU: {} ({:?}) backend={:?} format={:?}",
            adapter_info.name,
            adapter_info.device_type,
            adapter_info.backend,
            format
        );

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            surface: Some(surface),
            config: Some(config),
            size,
            target_format: format,
            srgb_target,
            adapter_info,
        })
    }

    pub async fn new_headless(width: u32, height: u32) -> Result<Self> {
        let instance = wgpu::Instance::default();
        let adapter = request_adapter(&instance, None).await?;
        let (device, queue) = create_device(&adapter).await?;

        let size = PhysicalSize::new(width.max(1), height.max(1));
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let adapter_info = adapter.get_info();

        log::info!(
            "Headless GPU: {} ({:?}) backend={:?}",
            adapter_info.name,
            adapter_info.device_type,
            adapter_info.backend
        );

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            surface: None,
            config: None,
            size,
            target_format: format,
            srgb_target: false,
            adapter_info,
        })
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        let width = new_size.width.max(1);
        let height = new_size.height.max(1);
        self.size = PhysicalSize::new(width, height);
        if let (Some(surface), Some(config)) = (&self.surface, &mut self.config) {
            config.width = width;
            config.height = height;
            surface.configure(&self.device, config);
        }
    }
}

async fn request_adapter(
    instance: &wgpu::Instance,
    surface: Option<&wgpu::Surface<'_>>,
) -> Result<wgpu::Adapter> {
    for force_fallback in [false, true] {
        if let Some(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: if force_fallback {
                    wgpu::PowerPreference::LowPower
                } else {
                    wgpu::PowerPreference::HighPerformance
                },
                force_fallback_adapter: force_fallback,
                compatible_surface: surface,
            })
            .await
        {
            return Ok(adapter);
        }
    }
    Err(anyhow!(
        "no suitable GPU adapter found. Install a Vulkan/Metal/DX12 driver or use a software rasterizer."
    ))
}

async fn create_device(adapter: &wgpu::Adapter) -> Result<(wgpu::Device, wgpu::Queue)> {
    let limits = gpu::gpu_limits_for(adapter);
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("kast-device"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )
        .await
        .context("failed to create wgpu device")?;
    Ok((device, queue))
}

fn choose_format(formats: &[wgpu::TextureFormat]) -> wgpu::TextureFormat {
    if let Some(f) = formats.iter().find(|f| !f.is_srgb()) {
        return *f;
    }
    formats
        .first()
        .copied()
        .unwrap_or(wgpu::TextureFormat::Bgra8Unorm)
}

fn choose_present_mode(modes: &[wgpu::PresentMode]) -> wgpu::PresentMode {
    if modes.contains(&wgpu::PresentMode::Mailbox) {
        wgpu::PresentMode::Mailbox
    } else if modes.contains(&wgpu::PresentMode::Fifo) {
        wgpu::PresentMode::Fifo
    } else {
        modes.first().copied().unwrap_or(wgpu::PresentMode::Fifo)
    }
}
