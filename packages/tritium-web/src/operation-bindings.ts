// Generated from spec/training/v2/manifest.json and vectors/v2.json.
// Run `npm run generate:bindings`; manual edits fail `npm run check`.

export const PORTABLE_OPERATION_BINDINGS_V1 = {
  "graph.ste_surrogate": {
    "forward": {
      "inputs": [
        "weight",
        "scale"
      ],
      "attributes": [
        {
          "name": "rows",
          "kind": "u64"
        },
        {
          "name": "cols",
          "kind": "u64"
        }
      ],
      "outputs": [
        "result"
      ]
    },
    "vjp": {
      "inputs": [
        "weight",
        "scale",
        "grad_output"
      ],
      "attributes": [
        {
          "name": "rows",
          "kind": "u64"
        },
        {
          "name": "cols",
          "kind": "u64"
        }
      ],
      "outputs": [
        "grad_weight",
        "grad_scale"
      ]
    }
  },
  "graph.salt_ste": {
    "forward": {
      "inputs": [
        "weight"
      ],
      "attributes": [
        {
          "name": "rows",
          "kind": "u64"
        },
        {
          "name": "cols",
          "kind": "u64"
        },
        {
          "name": "planes",
          "kind": "u64"
        }
      ],
      "outputs": [
        "result"
      ]
    },
    "vjp": {
      "inputs": [
        "weight",
        "grad_output"
      ],
      "attributes": [
        {
          "name": "rows",
          "kind": "u64"
        },
        {
          "name": "cols",
          "kind": "u64"
        },
        {
          "name": "planes",
          "kind": "u64"
        }
      ],
      "outputs": [
        "grad_weight"
      ]
    }
  },
  "graph.lsq_ste": {
    "forward": {
      "inputs": [
        "weight",
        "alpha"
      ],
      "attributes": [
        {
          "name": "rows",
          "kind": "u64"
        },
        {
          "name": "cols",
          "kind": "u64"
        }
      ],
      "outputs": [
        "result"
      ]
    },
    "vjp": {
      "inputs": [
        "weight",
        "alpha",
        "grad_output"
      ],
      "attributes": [
        {
          "name": "rows",
          "kind": "u64"
        },
        {
          "name": "cols",
          "kind": "u64"
        }
      ],
      "outputs": [
        "grad_weight",
        "grad_alpha"
      ]
    }
  },
  "graph.fsq": {
    "forward": {
      "inputs": [
        "x"
      ],
      "attributes": [
        {
          "name": "channels",
          "kind": "u64"
        },
        {
          "name": "len",
          "kind": "u64"
        },
        {
          "name": "levels",
          "kind": "u32-list"
        },
        {
          "name": "bound",
          "kind": "text"
        },
        {
          "name": "ste",
          "kind": "text"
        },
        {
          "name": "alpha",
          "kind": "f32"
        },
        {
          "name": "seed",
          "kind": "u64"
        }
      ],
      "outputs": [
        "result"
      ]
    },
    "vjp": {
      "inputs": [
        "x",
        "grad_output"
      ],
      "attributes": [
        {
          "name": "channels",
          "kind": "u64"
        },
        {
          "name": "len",
          "kind": "u64"
        },
        {
          "name": "levels",
          "kind": "u32-list"
        },
        {
          "name": "bound",
          "kind": "text"
        },
        {
          "name": "ste",
          "kind": "text"
        },
        {
          "name": "alpha",
          "kind": "f32"
        },
        {
          "name": "seed",
          "kind": "u64"
        }
      ],
      "outputs": [
        "grad_x"
      ]
    }
  },
  "graph.dense_matmul": {
    "forward": {
      "inputs": [
        "x",
        "weight"
      ],
      "attributes": [
        {
          "name": "m",
          "kind": "u64"
        },
        {
          "name": "n",
          "kind": "u64"
        },
        {
          "name": "k",
          "kind": "u64"
        }
      ],
      "outputs": [
        "result"
      ]
    },
    "vjp": {
      "inputs": [
        "x",
        "weight",
        "grad_output"
      ],
      "attributes": [
        {
          "name": "m",
          "kind": "u64"
        },
        {
          "name": "n",
          "kind": "u64"
        },
        {
          "name": "k",
          "kind": "u64"
        }
      ],
      "outputs": [
        "grad_x",
        "grad_weight"
      ]
    }
  },
  "graph.ternary_matmul": {
    "forward": {
      "inputs": [
        "activation",
        "weight",
        "scale"
      ],
      "attributes": [
        {
          "name": "m",
          "kind": "u64"
        },
        {
          "name": "n",
          "kind": "u64"
        },
        {
          "name": "k",
          "kind": "u64"
        }
      ],
      "outputs": [
        "result"
      ]
    },
    "vjp": {
      "inputs": [
        "activation",
        "weight",
        "scale",
        "grad_output"
      ],
      "attributes": [
        {
          "name": "m",
          "kind": "u64"
        },
        {
          "name": "n",
          "kind": "u64"
        },
        {
          "name": "k",
          "kind": "u64"
        }
      ],
      "outputs": [
        "grad_activation",
        "grad_weight",
        "grad_scale"
      ]
    }
  },
  "graph.transpose": {
    "forward": {
      "inputs": [
        "x"
      ],
      "attributes": [
        {
          "name": "rows",
          "kind": "u64"
        },
        {
          "name": "cols",
          "kind": "u64"
        }
      ],
      "outputs": [
        "result"
      ]
    },
    "vjp": {
      "inputs": [
        "grad_output"
      ],
      "attributes": [
        {
          "name": "rows",
          "kind": "u64"
        },
        {
          "name": "cols",
          "kind": "u64"
        }
      ],
      "outputs": [
        "grad_x"
      ]
    }
  },
  "graph.embedding_gather": {
    "forward": {
      "inputs": [
        "weight",
        "tokens"
      ],
      "attributes": [
        {
          "name": "vocab",
          "kind": "u64"
        },
        {
          "name": "n_embd",
          "kind": "u64"
        }
      ],
      "outputs": [
        "result"
      ]
    },
    "vjp": {
      "inputs": [
        "weight",
        "tokens",
        "grad_output"
      ],
      "attributes": [
        {
          "name": "vocab",
          "kind": "u64"
        },
        {
          "name": "n_embd",
          "kind": "u64"
        }
      ],
      "outputs": [
        "grad_weight"
      ]
    }
  },
  "graph.slice_cols": {
    "forward": {
      "inputs": [
        "x"
      ],
      "attributes": [
        {
          "name": "rows",
          "kind": "u64"
        },
        {
          "name": "cols",
          "kind": "u64"
        },
        {
          "name": "start",
          "kind": "u64"
        },
        {
          "name": "len",
          "kind": "u64"
        }
      ],
      "outputs": [
        "result"
      ]
    },
    "vjp": {
      "inputs": [
        "grad_output"
      ],
      "attributes": [
        {
          "name": "rows",
          "kind": "u64"
        },
        {
          "name": "cols",
          "kind": "u64"
        },
        {
          "name": "start",
          "kind": "u64"
        },
        {
          "name": "len",
          "kind": "u64"
        }
      ],
      "outputs": [
        "grad_x"
      ]
    }
  },
  "graph.concat_cols": {
    "forward": {
      "inputs": [
        "part.0",
        "part.1"
      ],
      "attributes": [
        {
          "name": "rows",
          "kind": "u64"
        },
        {
          "name": "lens",
          "kind": "u64-list"
        }
      ],
      "outputs": [
        "result"
      ]
    },
    "vjp": {
      "inputs": [
        "grad_output"
      ],
      "attributes": [
        {
          "name": "rows",
          "kind": "u64"
        },
        {
          "name": "lens",
          "kind": "u64-list"
        }
      ],
      "outputs": [
        "grad_part.0",
        "grad_part.1"
      ]
    }
  },
  "graph.detach": {
    "forward": {
      "inputs": [
        "x"
      ],
      "attributes": [],
      "outputs": [
        "result"
      ]
    },
    "vjp": {
      "inputs": [
        "grad_output"
      ],
      "attributes": [],
      "outputs": [
        "grad_x"
      ]
    }
  },
  "graph.scale_const": {
    "forward": {
      "inputs": [
        "x"
      ],
      "attributes": [
        {
          "name": "scale",
          "kind": "f32"
        }
      ],
      "outputs": [
        "result"
      ]
    },
    "vjp": {
      "inputs": [
        "grad_output"
      ],
      "attributes": [
        {
          "name": "scale",
          "kind": "f32"
        }
      ],
      "outputs": [
        "grad_x"
      ]
    }
  },
  "graph.bias": {
    "forward": {
      "inputs": [
        "x",
        "bias"
      ],
      "attributes": [
        {
          "name": "rows",
          "kind": "u64"
        },
        {
          "name": "cols",
          "kind": "u64"
        }
      ],
      "outputs": [
        "result"
      ]
    },
    "vjp": {
      "inputs": [
        "x",
        "bias",
        "grad_output"
      ],
      "attributes": [
        {
          "name": "rows",
          "kind": "u64"
        },
        {
          "name": "cols",
          "kind": "u64"
        }
      ],
      "outputs": [
        "grad_x",
        "grad_bias"
      ]
    }
  },
  "graph.add": {
    "forward": {
      "inputs": [
        "left",
        "right"
      ],
      "attributes": [],
      "outputs": [
        "result"
      ]
    },
    "vjp": {
      "inputs": [
        "grad_output"
      ],
      "attributes": [],
      "outputs": [
        "grad_left",
        "grad_right"
      ]
    }
  },
  "graph.mul": {
    "forward": {
      "inputs": [
        "left",
        "right"
      ],
      "attributes": [],
      "outputs": [
        "result"
      ]
    },
    "vjp": {
      "inputs": [
        "left",
        "right",
        "grad_output"
      ],
      "attributes": [],
      "outputs": [
        "grad_left",
        "grad_right"
      ]
    }
  },
  "graph.conv1d": {
    "forward": {
      "inputs": [
        "x",
        "weight",
        "scale"
      ],
      "attributes": [
        {
          "name": "batch",
          "kind": "u64"
        },
        {
          "name": "c_in",
          "kind": "u64"
        },
        {
          "name": "c_out",
          "kind": "u64"
        },
        {
          "name": "l_in",
          "kind": "u64"
        },
        {
          "name": "k",
          "kind": "u64"
        },
        {
          "name": "stride",
          "kind": "u64"
        },
        {
          "name": "dilation",
          "kind": "u64"
        },
        {
          "name": "pad_left",
          "kind": "u64"
        },
        {
          "name": "pad_right",
          "kind": "u64"
        },
        {
          "name": "groups",
          "kind": "u64"
        }
      ],
      "outputs": [
        "result"
      ]
    },
    "vjp": {
      "inputs": [
        "x",
        "weight",
        "scale",
        "grad_output"
      ],
      "attributes": [
        {
          "name": "batch",
          "kind": "u64"
        },
        {
          "name": "c_in",
          "kind": "u64"
        },
        {
          "name": "c_out",
          "kind": "u64"
        },
        {
          "name": "l_in",
          "kind": "u64"
        },
        {
          "name": "k",
          "kind": "u64"
        },
        {
          "name": "stride",
          "kind": "u64"
        },
        {
          "name": "dilation",
          "kind": "u64"
        },
        {
          "name": "pad_left",
          "kind": "u64"
        },
        {
          "name": "pad_right",
          "kind": "u64"
        },
        {
          "name": "groups",
          "kind": "u64"
        }
      ],
      "outputs": [
        "grad_x",
        "grad_weight",
        "grad_scale"
      ]
    }
  },
  "graph.conv2d": {
    "forward": {
      "inputs": [
        "x",
        "weight",
        "scale"
      ],
      "attributes": [
        {
          "name": "batch",
          "kind": "u64"
        },
        {
          "name": "c_in",
          "kind": "u64"
        },
        {
          "name": "c_out",
          "kind": "u64"
        },
        {
          "name": "input_h",
          "kind": "u64"
        },
        {
          "name": "input_w",
          "kind": "u64"
        },
        {
          "name": "kernel_h",
          "kind": "u64"
        },
        {
          "name": "kernel_w",
          "kind": "u64"
        },
        {
          "name": "stride_h",
          "kind": "u64"
        },
        {
          "name": "stride_w",
          "kind": "u64"
        },
        {
          "name": "dilation_h",
          "kind": "u64"
        },
        {
          "name": "dilation_w",
          "kind": "u64"
        },
        {
          "name": "pad_top",
          "kind": "u64"
        },
        {
          "name": "pad_bottom",
          "kind": "u64"
        },
        {
          "name": "pad_left",
          "kind": "u64"
        },
        {
          "name": "pad_right",
          "kind": "u64"
        },
        {
          "name": "groups",
          "kind": "u64"
        }
      ],
      "outputs": [
        "result"
      ]
    },
    "vjp": {
      "inputs": [
        "x",
        "weight",
        "scale",
        "grad_output"
      ],
      "attributes": [
        {
          "name": "batch",
          "kind": "u64"
        },
        {
          "name": "c_in",
          "kind": "u64"
        },
        {
          "name": "c_out",
          "kind": "u64"
        },
        {
          "name": "input_h",
          "kind": "u64"
        },
        {
          "name": "input_w",
          "kind": "u64"
        },
        {
          "name": "kernel_h",
          "kind": "u64"
        },
        {
          "name": "kernel_w",
          "kind": "u64"
        },
        {
          "name": "stride_h",
          "kind": "u64"
        },
        {
          "name": "stride_w",
          "kind": "u64"
        },
        {
          "name": "dilation_h",
          "kind": "u64"
        },
        {
          "name": "dilation_w",
          "kind": "u64"
        },
        {
          "name": "pad_top",
          "kind": "u64"
        },
        {
          "name": "pad_bottom",
          "kind": "u64"
        },
        {
          "name": "pad_left",
          "kind": "u64"
        },
        {
          "name": "pad_right",
          "kind": "u64"
        },
        {
          "name": "groups",
          "kind": "u64"
        }
      ],
      "outputs": [
        "grad_x",
        "grad_weight",
        "grad_scale"
      ]
    }
  },
  "graph.relu2": {
    "forward": {
      "inputs": [
        "x"
      ],
      "attributes": [],
      "outputs": [
        "result"
      ]
    },
    "vjp": {
      "inputs": [
        "x",
        "grad_output"
      ],
      "attributes": [],
      "outputs": [
        "grad_x"
      ]
    }
  },
  "graph.silu": {
    "forward": {
      "inputs": [
        "x"
      ],
      "attributes": [],
      "outputs": [
        "result"
      ]
    },
    "vjp": {
      "inputs": [
        "x",
        "grad_output"
      ],
      "attributes": [],
      "outputs": [
        "grad_x"
      ]
    }
  },
  "graph.rmsnorm": {
    "forward": {
      "inputs": [
        "x",
        "weight"
      ],
      "attributes": [
        {
          "name": "rows",
          "kind": "u64"
        },
        {
          "name": "cols",
          "kind": "u64"
        },
        {
          "name": "eps",
          "kind": "f32"
        }
      ],
      "outputs": [
        "result"
      ]
    },
    "vjp": {
      "inputs": [
        "x",
        "weight",
        "grad_output"
      ],
      "attributes": [
        {
          "name": "rows",
          "kind": "u64"
        },
        {
          "name": "cols",
          "kind": "u64"
        },
        {
          "name": "eps",
          "kind": "f32"
        }
      ],
      "outputs": [
        "grad_x",
        "grad_weight"
      ]
    }
  },
  "graph.softmax": {
    "forward": {
      "inputs": [
        "x"
      ],
      "attributes": [
        {
          "name": "rows",
          "kind": "u64"
        },
        {
          "name": "cols",
          "kind": "u64"
        }
      ],
      "outputs": [
        "result"
      ]
    },
    "vjp": {
      "inputs": [
        "x",
        "grad_output"
      ],
      "attributes": [
        {
          "name": "rows",
          "kind": "u64"
        },
        {
          "name": "cols",
          "kind": "u64"
        }
      ],
      "outputs": [
        "grad_x"
      ]
    }
  },
  "graph.causal_mask": {
    "forward": {
      "inputs": [
        "x"
      ],
      "attributes": [
        {
          "name": "rows",
          "kind": "u64"
        },
        {
          "name": "cols",
          "kind": "u64"
        }
      ],
      "outputs": [
        "result"
      ]
    },
    "vjp": {
      "inputs": [
        "grad_output"
      ],
      "attributes": [
        {
          "name": "rows",
          "kind": "u64"
        },
        {
          "name": "cols",
          "kind": "u64"
        }
      ],
      "outputs": [
        "grad_x"
      ]
    }
  },
  "graph.rope": {
    "forward": {
      "inputs": [
        "x"
      ],
      "attributes": [
        {
          "name": "positions",
          "kind": "u64-list"
        },
        {
          "name": "n_head",
          "kind": "u64"
        },
        {
          "name": "head_dim",
          "kind": "u64"
        },
        {
          "name": "theta",
          "kind": "f32"
        }
      ],
      "outputs": [
        "result"
      ]
    },
    "vjp": {
      "inputs": [
        "grad_output"
      ],
      "attributes": [
        {
          "name": "positions",
          "kind": "u64-list"
        },
        {
          "name": "n_head",
          "kind": "u64"
        },
        {
          "name": "head_dim",
          "kind": "u64"
        },
        {
          "name": "theta",
          "kind": "f32"
        }
      ],
      "outputs": [
        "grad_x"
      ]
    }
  },
  "graph.attention": {
    "forward": {
      "inputs": [
        "q",
        "k",
        "v"
      ],
      "attributes": [
        {
          "name": "seq",
          "kind": "u64"
        },
        {
          "name": "n_head",
          "kind": "u64"
        },
        {
          "name": "n_kv_head",
          "kind": "u64"
        },
        {
          "name": "head_dim",
          "kind": "u64"
        },
        {
          "name": "causal",
          "kind": "bool"
        }
      ],
      "outputs": [
        "result"
      ]
    },
    "vjp": {
      "inputs": [
        "q",
        "k",
        "v",
        "grad_output"
      ],
      "attributes": [
        {
          "name": "seq",
          "kind": "u64"
        },
        {
          "name": "n_head",
          "kind": "u64"
        },
        {
          "name": "n_kv_head",
          "kind": "u64"
        },
        {
          "name": "head_dim",
          "kind": "u64"
        },
        {
          "name": "causal",
          "kind": "bool"
        }
      ],
      "outputs": [
        "grad_q",
        "grad_k",
        "grad_v"
      ]
    }
  },
  "loss.mse": {
    "forward": {
      "inputs": [
        "prediction",
        "target"
      ],
      "attributes": [],
      "outputs": [
        "result"
      ]
    },
    "vjp": {
      "inputs": [
        "prediction",
        "target",
        "grad_output"
      ],
      "attributes": [],
      "outputs": [
        "grad_prediction"
      ]
    }
  },
  "loss.softmax_cross_entropy": {
    "forward": {
      "inputs": [
        "logits",
        "target"
      ],
      "attributes": [
        {
          "name": "rows",
          "kind": "u64"
        },
        {
          "name": "cols",
          "kind": "u64"
        }
      ],
      "outputs": [
        "result"
      ]
    },
    "vjp": {
      "inputs": [
        "logits",
        "target",
        "grad_output"
      ],
      "attributes": [
        {
          "name": "rows",
          "kind": "u64"
        },
        {
          "name": "cols",
          "kind": "u64"
        }
      ],
      "outputs": [
        "grad_logits"
      ]
    }
  },
  "loss.topk_knowledge_distillation": {
    "forward": {
      "inputs": [
        "logits",
        "indices",
        "probabilities"
      ],
      "attributes": [
        {
          "name": "rows",
          "kind": "u64"
        },
        {
          "name": "cols",
          "kind": "u64"
        },
        {
          "name": "k",
          "kind": "u64"
        }
      ],
      "outputs": [
        "result"
      ]
    },
    "vjp": {
      "inputs": [
        "logits",
        "indices",
        "probabilities",
        "grad_output"
      ],
      "attributes": [
        {
          "name": "rows",
          "kind": "u64"
        },
        {
          "name": "cols",
          "kind": "u64"
        },
        {
          "name": "k",
          "kind": "u64"
        }
      ],
      "outputs": [
        "grad_logits"
      ]
    }
  },
  "optimizer.sgd": {
    "step": {
      "inputs": [
        "parameter",
        "gradient"
      ],
      "attributes": [
        {
          "name": "step",
          "kind": "u64"
        },
        {
          "name": "lr",
          "kind": "f32"
        }
      ],
      "outputs": [
        "parameter"
      ]
    }
  },
  "optimizer.adamw": {
    "step": {
      "inputs": [
        "parameter",
        "gradient",
        "moment1",
        "moment2"
      ],
      "attributes": [
        {
          "name": "step",
          "kind": "u64"
        },
        {
          "name": "lr",
          "kind": "f32"
        },
        {
          "name": "beta1",
          "kind": "f32"
        },
        {
          "name": "beta2",
          "kind": "f32"
        },
        {
          "name": "eps",
          "kind": "f32"
        },
        {
          "name": "weight_decay",
          "kind": "f32"
        }
      ],
      "outputs": [
        "parameter",
        "moment1",
        "moment2"
      ]
    }
  },
  "optimizer.cautious_adamw": {
    "step": {
      "inputs": [
        "parameter",
        "gradient",
        "moment1",
        "moment2"
      ],
      "attributes": [
        {
          "name": "step",
          "kind": "u64"
        },
        {
          "name": "lr",
          "kind": "f32"
        },
        {
          "name": "beta1",
          "kind": "f32"
        },
        {
          "name": "beta2",
          "kind": "f32"
        },
        {
          "name": "eps",
          "kind": "f32"
        },
        {
          "name": "weight_decay",
          "kind": "f32"
        }
      ],
      "outputs": [
        "parameter",
        "moment1",
        "moment2"
      ]
    }
  },
  "optimizer.int8_adamw": {
    "step": {
      "inputs": [
        "parameter",
        "gradient",
        "moment1_q8",
        "moment2_q8",
        "moment1_scale",
        "moment2_scale"
      ],
      "attributes": [
        {
          "name": "step",
          "kind": "u64"
        },
        {
          "name": "lr",
          "kind": "f32"
        },
        {
          "name": "beta1",
          "kind": "f32"
        },
        {
          "name": "beta2",
          "kind": "f32"
        },
        {
          "name": "eps",
          "kind": "f32"
        },
        {
          "name": "weight_decay",
          "kind": "f32"
        }
      ],
      "outputs": [
        "parameter",
        "moment1_q8",
        "moment2_q8",
        "moment1_scale",
        "moment2_scale"
      ]
    }
  },
  "optimizer.muon": {
    "step": {
      "inputs": [
        "parameter",
        "gradient",
        "momentum"
      ],
      "attributes": [
        {
          "name": "step",
          "kind": "u64"
        },
        {
          "name": "lr",
          "kind": "f32"
        },
        {
          "name": "momentum",
          "kind": "f32"
        },
        {
          "name": "weight_decay",
          "kind": "f32"
        },
        {
          "name": "rows",
          "kind": "u64"
        },
        {
          "name": "cols",
          "kind": "u64"
        },
        {
          "name": "ns_steps",
          "kind": "u64"
        }
      ],
      "outputs": [
        "parameter",
        "momentum"
      ]
    }
  }
} as const;
