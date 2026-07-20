struct Params {
    rows: u32,
    part_count: u32,
    total_columns: u32,
    padding: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> values: array<f32>;
@group(0) @binding(2) var<storage, read> lengths: array<u32>;
@group(0) @binding(3) var<storage, read> offsets: array<u32>;
@group(0) @binding(4) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= params.rows * params.total_columns) {
        return;
    }
    let row = index / params.total_columns;
    let output_column = index % params.total_columns;
    var column_start = 0u;
    for (var part = 0u; part < params.part_count; part = part + 1u) {
        let width = lengths[part];
        if (output_column < column_start + width) {
            let local_column = output_column - column_start;
            result[index] = values[offsets[part] + row * width + local_column];
            return;
        }
        column_start = column_start + width;
    }
}
