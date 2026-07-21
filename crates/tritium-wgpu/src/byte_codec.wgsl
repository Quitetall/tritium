struct Params {
    len: u32,
    padding_0: u32,
    padding_1: u32,
    padding_2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> source: array<u32>;
@group(0) @binding(2) var<storage, read_write> destination: array<u32>;

@compute @workgroup_size(64, 1, 1)
fn unpack(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index < params.len) {
        let word = source[index / 4u];
        destination[index] = (word >> ((index % 4u) * 8u)) & 255u;
    }
}

@compute @workgroup_size(64, 1, 1)
fn pack(@builtin(global_invocation_id) gid: vec3<u32>) {
    let word_index = gid.x;
    let words = params.len / 4u + select(0u, 1u, params.len % 4u != 0u);
    if (word_index < words) {
        let first = word_index * 4u;
        var word = 0u;
        for (var lane = 0u; lane < 4u; lane += 1u) {
            let index = first + lane;
            if (index < params.len) {
                word |= (source[index] & 255u) << (lane * 8u);
            }
        }
        destination[word_index] = word;
    }
}
