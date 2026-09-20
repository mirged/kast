pub fn create(device: &wgpu::Device, label: &str, source: &str) -> wgpu::ShaderModule {
    let combined = format!("{}\n{}", include_str!("../../shaders/common.wgsl"), source);
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(combined.into()),
    })
}
