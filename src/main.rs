mod app;
use kast::{gpu, scene};

use anyhow::{Context, Result};
use std::path::PathBuf;
use winit::event_loop::{ControlFlow, EventLoop};

use kast::camera::Camera;
use kast::context::GpuContext;
use kast::renderer::Renderer;
use kast::scene::loader;

struct Args {
    model: Option<PathBuf>,
    headless: bool,
    screenshot: Option<PathBuf>,
    frames: u32,
    width: u32,
    height: u32,
    no_ssao: bool,
    no_ssr: bool,
    no_shadows: bool,
    no_ibl: bool,
    no_bloom: bool,
    dof: bool,
}

fn parse_args() -> Args {
    let mut args = Args {
        model: None,
        headless: false,
        screenshot: None,
        frames: 30,
        width: 1280,
        height: 720,
        no_ssao: false,
        no_ssr: false,
        no_shadows: false,
        no_ibl: false,
        no_bloom: false,
        dof: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--model" | "-m" => args.model = it.next().map(PathBuf::from),
            "--headless" => args.headless = true,
            "--screenshot" | "-o" => {
                args.screenshot = it.next().map(PathBuf::from);
                args.headless = true;
            }
            "--frames" | "-n" => {
                if let Some(v) = it.next().and_then(|v| v.parse().ok()) {
                    args.frames = v;
                }
            }
            "--width" => {
                if let Some(v) = it.next().and_then(|v| v.parse().ok()) {
                    args.width = v;
                }
            }
            "--height" => {
                if let Some(v) = it.next().and_then(|v| v.parse().ok()) {
                    args.height = v;
                }
            }
            "--no-ssao" => args.no_ssao = true,
            "--no-ssr" => args.no_ssr = true,
            "--no-shadows" => args.no_shadows = true,
            "--no-ibl" => args.no_ibl = true,
            "--no-bloom" => args.no_bloom = true,
            "--dof" => args.dof = true,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown argument: {other}");
                print_help();
                std::process::exit(1);
            }
        }
    }
    args
}

fn print_help() {
    println!(
        "kast - wgpu photorealistic renderer\n\n\
         USAGE:\n  kast [OPTIONS]\n\n\
         OPTIONS:\n\
         \x20 -m, --model <PATH>       Load a glTF/GLB model instead of the demo scene\n\
         \x20     --headless           Render without a window\n\
         \x20 -o, --screenshot <PATH>  Render a frame and save it as PNG (implies --headless)\n\
         \x20 -n, --frames <N>         Number of frames to render in headless mode (default 30)\n\
         \x20     --width <PIXELS>      Offscreen width (default 1280)\n\
         \x20     --height <PIXELS>     Offscreen height (default 720)\n\n\
         CONTROLS:\n\
         \x20 RMB drag   rotate camera (orbit/fly)\n\
         \x20 MMB drag   pan\n\
         \x20 wheel      zoom (orbit) / speed (fly)\n\
         \x20 W A S D    move\n\
         \x20 Q E        down / up\n\
         \x20 Tab        toggle orbit/fly\n\
         \x20 [ ]        decrease / increase exposure\n\
         \x20 F1..F6    toggle SSR / SSAO / shadows / IBL / DoF / FXAA\n\
         \x20 F7         cycle tonemapper\n\
         \x20 Esc        quit"
    );
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(
        env_logger::Env::default()
            .default_filter_or("kast=info,wgpu_core=warn,wgpu_hal=warn,naga=warn,wgpu=warn"),
    )
    .init();
    let args = parse_args();

    if args.headless {
        run_headless(args)
    } else {
        run_windowed(args.model)
    }
}

fn run_windowed(model: Option<PathBuf>) -> Result<()> {
    let event_loop = EventLoop::new().context("failed to create event loop")?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = app::App::new(model);
    event_loop.run_app(&mut app).context("event loop failed")?;
    Ok(())
}

fn run_headless(args: Args) -> Result<()> {
    log::info!("headless mode: {}x{}", args.width, args.height);
    let ctx = pollster::block_on(GpuContext::new_headless(args.width, args.height))?;

    let mut scene = match &args.model {
        Some(path) => loader::load(path, &ctx.device, &ctx.queue)?,
        None => scene::default_scene(&ctx.device, &ctx.queue)?,
    };
    let mut camera = Camera::default();
    camera.target = glam::Vec3::new(0.0, 1.0, 0.0);
    camera.distance = 9.0;

    let mut renderer = Renderer::new(
        &ctx.device,
        &ctx.queue,
        ctx.size.width,
        ctx.size.height,
        ctx.target_format,
        ctx.srgb_target,
        true,
        &mut scene,
    )?;

    renderer.settings.ssao_enabled &= !args.no_ssao;
    renderer.settings.ssr_enabled &= !args.no_ssr;
    renderer.settings.shadows_enabled &= !args.no_shadows;
    renderer.settings.ibl_enabled &= !args.no_ibl;
    if args.no_bloom {
        renderer.settings.bloom_intensity = 0.0;
    }
    renderer.settings.dof_enabled = args.dof;

    let frames = args.frames.max(1);
    for i in 0..frames {
        let angle = i as f32 / frames as f32 * std::f32::consts::TAU;
        camera.yaw = angle * 0.0 - 0.6;
        camera.update(1.0 / 60.0);
        renderer.render(&ctx.device, &ctx.queue, &scene, &camera, None, 1.0 / 60.0);
    }

    ctx.device.poll(wgpu::Maintain::Wait);

    if let Some(path) = &args.screenshot {
        let present = renderer.present.as_ref().expect("headless present target");
        let pixels = readback(
            &ctx.device,
            &ctx.queue,
            &present.texture,
            present.width,
            present.height,
        );
        image::save_buffer(
            path,
            &pixels,
            present.width,
            present.height,
            image::ColorType::Rgba8,
        )
        .with_context(|| format!("failed to save screenshot to {}", path.display()))?;
        log::info!("wrote {}", path.display());
    } else {
        log::info!("rendered {frames} frames (no screenshot requested)");
    }

    Ok(())
}

fn readback(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
) -> Vec<u8> {
    let unpadded = width * 4;
    let padded = gpu::align_to(unpadded, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (padded * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("readback-encoder"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &buffer,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));

    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device.poll(wgpu::Maintain::Wait);
    let _ = rx.recv();

    let data = slice.get_mapped_range();
    let mut out = Vec::with_capacity((unpadded * height) as usize);
    for row in 0..height {
        let start = (row * padded) as usize;
        out.extend_from_slice(&data[start..start + unpadded as usize]);
    }
    drop(data);
    buffer.unmap();
    out
}
