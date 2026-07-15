//! Sharded, content-addressed activation cache for the SALT V2 calibration pipeline.

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(seed: u8) -> ActivationDigest {
        ActivationDigest::from_bytes([seed; 32])
    }

    fn spec(
        dtype: ActivationDType,
        tokens: u64,
        features: u64,
        shard_tokens: u32,
    ) -> ActivationCacheSpec {
        ActivationCacheSpec::new(
            7,
            "model.layers.7.mlp.down_proj.input",
            tokens,
            features,
            dtype,
            digest(9),
            shard_tokens,
        )
        .expect("valid fixture spec")
    }

    fn chunk(
        spec: &ActivationCacheSpec,
        token_start: u64,
        token_count: u64,
        values: Vec<f32>,
        token_mask: Vec<bool>,
        sequence_ends: Vec<u64>,
    ) -> ActivationChunk {
        ActivationChunk::new(
            spec,
            token_start,
            token_count,
            values,
            token_mask,
            sequence_ends,
        )
        .expect("valid fixture chunk")
    }

    fn finalized_cache() -> ActivationCache {
        let spec = spec(ActivationDType::Float16, 5, 2, 2);
        let mut builder = ActivationCacheBuilder::new(spec.clone());
        builder
            .ingest(chunk(
                &spec,
                0,
                5,
                vec![1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 5.0, -5.0],
                vec![true, false, true, true, false],
                vec![2, 5],
            ))
            .expect("ingest reopen fixture");
        builder.finalize().expect("finalize reopen fixture")
    }

    #[test]
    fn canonical_bytes_reopen_to_the_exact_same_cache_and_identity() {
        let original = finalized_cache();
        let reopened = ActivationCache::from_encoded(original.encoded()).expect("reopen cache");

        assert_eq!(reopened, original);
        assert_eq!(reopened.digest(), original.digest());
        assert_eq!(reopened.encoded(), original.encoded());
        assert_eq!(reopened.byte_ledger(), original.byte_ledger());
        assert_eq!(reopened.shards(), original.shards());
    }

    #[test]
    fn bounded_reader_supports_durable_resume_and_rejects_oversize_input() {
        let original = finalized_cache();
        let reopened = ActivationCache::read_from(
            std::io::Cursor::new(original.encoded()),
            original.byte_ledger().encoded_bytes(),
        )
        .expect("read persisted cache");
        assert_eq!(reopened, original);

        assert!(
            ActivationCache::read_from(
                std::io::Cursor::new(original.encoded()),
                original.byte_ledger().encoded_bytes() - 1,
            )
            .is_err()
        );
    }

    #[test]
    fn reopen_rejects_corruption_trailing_bytes_noncanonical_masks_and_digest_mismatch() {
        let original = finalized_cache();

        let mut bad_magic = original.encoded().to_vec();
        bad_magic[0] ^= 0x80;
        assert!(ActivationCache::from_encoded(&bad_magic).is_err());

        let mut bad_version = original.encoded().to_vec();
        bad_version[4..6].copy_from_slice(&2_u16.to_le_bytes());
        assert!(ActivationCache::from_encoded(&bad_version).is_err());

        let mut bad_reserved = original.encoded().to_vec();
        bad_reserved[7] = 1;
        assert!(ActivationCache::from_encoded(&bad_reserved).is_err());

        let mut bad_dtype = original.encoded().to_vec();
        bad_dtype[6] = 0;
        assert!(ActivationCache::from_encoded(&bad_dtype).is_err());

        let first = &original.shards()[0];
        let value_offset = usize::try_from(first.encoded_offset()).expect("offset")
            + usize::try_from(SHARD_HEADER_BYTES).expect("header");
        let mut nonfinite_value = original.encoded().to_vec();
        nonfinite_value[value_offset..value_offset + 2]
            .copy_from_slice(&f16::INFINITY.to_bits().to_le_bytes());
        assert!(ActivationCache::from_encoded(&nonfinite_value).is_err());

        let mut changed_finite_value = original.encoded().to_vec();
        changed_finite_value[value_offset] ^= 1;
        assert!(
            ActivationCache::from_encoded_verified(&changed_finite_value, original.digest())
                .is_err()
        );

        let mask_offset = usize::try_from(first.encoded_offset()).expect("offset")
            + usize::try_from(SHARD_HEADER_BYTES).expect("header")
            + usize::try_from(first.value_bytes()).expect("values");
        let mut bad_mask = original.encoded().to_vec();
        bad_mask[mask_offset] |= 0b1111_1100;
        assert!(ActivationCache::from_encoded(&bad_mask).is_err());

        let mut trailing = original.encoded().to_vec();
        trailing.push(0);
        assert!(ActivationCache::from_encoded(&trailing).is_err());

        assert!(ActivationCache::from_encoded_verified(original.encoded(), digest(77)).is_err());
    }

    #[test]
    fn reopen_rejects_oversized_counts_and_duplicate_or_gapped_shards_before_allocation() {
        let original = finalized_cache();

        let mut oversized_tokens = original.encoded().to_vec();
        oversized_tokens[12..20].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(ActivationCache::from_encoded(&oversized_tokens).is_err());

        let second_offset =
            usize::try_from(original.shards()[1].encoded_offset()).expect("second shard offset");
        let mut duplicate_index = original.encoded().to_vec();
        duplicate_index[second_offset..second_offset + 4].copy_from_slice(&0_u32.to_le_bytes());
        assert!(ActivationCache::from_encoded(&duplicate_index).is_err());

        let mut gapped_range = original.encoded().to_vec();
        let wrong_start = original.shards()[1].token_start() + 1;
        gapped_range[second_offset + 4..second_offset + 12]
            .copy_from_slice(&wrong_start.to_le_bytes());
        assert!(ActivationCache::from_encoded(&gapped_range).is_err());
    }

    #[test]
    fn arbitrary_short_inputs_are_rejected_without_panicking_or_allocating_from_counts() {
        for len in 0..128_usize {
            let bytes: Vec<u8> = (0..len)
                .map(|index| (index as u8).wrapping_mul(73).wrapping_add(len as u8))
                .collect();
            assert!(ActivationCache::from_encoded(&bytes).is_err(), "len={len}");
        }
    }

    #[test]
    fn artifact_digest_is_independent_of_ingestion_order_and_chunking() {
        let spec = spec(ActivationDType::Float32, 6, 2, 2);
        let values: Vec<f32> = (0..12).map(|value| value as f32 / 4.0).collect();
        let mask = vec![true, true, true, false, true, true];

        let mut one_shot = ActivationCacheBuilder::new(spec.clone());
        one_shot
            .ingest(chunk(
                &spec,
                0,
                6,
                values.clone(),
                mask.clone(),
                vec![2, 5, 6],
            ))
            .expect("ingest one-shot chunk");
        let one_shot = one_shot.finalize().expect("finalize one-shot cache");

        let mut resumed_left = ActivationCacheBuilder::new(spec.clone());
        resumed_left
            .ingest(chunk(
                &spec,
                4,
                2,
                values[8..].to_vec(),
                mask[4..].to_vec(),
                vec![5, 6],
            ))
            .expect("ingest tail first");
        let mut resumed_right = ActivationCacheBuilder::new(spec.clone());
        resumed_right
            .ingest(chunk(
                &spec,
                2,
                2,
                values[4..8].to_vec(),
                mask[2..4].to_vec(),
                Vec::new(),
            ))
            .expect("ingest middle");
        resumed_right
            .ingest(chunk(
                &spec,
                0,
                2,
                values[..4].to_vec(),
                mask[..2].to_vec(),
                vec![2],
            ))
            .expect("ingest head last");
        resumed_left
            .merge(resumed_right)
            .expect("merge resumed builders");
        let resumed = resumed_left.finalize().expect("finalize resumed cache");

        assert_eq!(resumed.digest(), one_shot.digest());
        assert_eq!(resumed.encoded(), one_shot.encoded());
        assert_eq!(resumed.sequence_ends(), &[2, 5, 6]);
    }

    #[test]
    fn source_layer_tensor_shape_and_dtype_are_digest_bound() {
        let base = spec(ActivationDType::Float32, 2, 2, 2);
        let base_chunk = chunk(
            &base,
            0,
            2,
            vec![1.0, 2.0, 3.0, 4.0],
            vec![true, true],
            vec![2],
        );

        let variants = [
            ActivationCacheSpec::new(
                8,
                base.tensor_name(),
                2,
                2,
                ActivationDType::Float32,
                digest(9),
                2,
            )
            .expect("layer variant"),
            ActivationCacheSpec::new(
                7,
                "other.tensor",
                2,
                2,
                ActivationDType::Float32,
                digest(9),
                2,
            )
            .expect("tensor variant"),
            spec(ActivationDType::Float32, 2, 3, 2),
            spec(ActivationDType::BFloat16, 2, 2, 2),
            ActivationCacheSpec::new(
                7,
                base.tensor_name(),
                2,
                2,
                ActivationDType::Float32,
                digest(10),
                2,
            )
            .expect("source variant"),
        ];

        for variant in variants {
            assert_ne!(variant.schema_digest(), base.schema_digest());
            let mut builder = ActivationCacheBuilder::new(variant);
            assert!(matches!(
                builder.ingest(base_chunk.clone()),
                Err(ActivationCacheError::SchemaMismatch { .. })
            ));
        }
    }

    #[test]
    fn artifact_digest_binds_value_order_mask_boundaries_and_provenance() {
        fn finish(
            spec: &ActivationCacheSpec,
            values: Vec<f32>,
            mask: Vec<bool>,
            boundaries: Vec<u64>,
        ) -> ActivationDigest {
            let mut builder = ActivationCacheBuilder::new(spec.clone());
            builder
                .ingest(chunk(spec, 0, 4, values, mask, boundaries))
                .expect("ingest digest fixture");
            builder
                .finalize()
                .expect("finalize digest fixture")
                .digest()
        }

        let base_spec = spec(ActivationDType::Float32, 4, 1, 2);
        let base = finish(
            &base_spec,
            vec![1.0, 2.0, 3.0, 4.0],
            vec![true, true, false, true],
            vec![2, 4],
        );
        let reordered = finish(
            &base_spec,
            vec![2.0, 1.0, 3.0, 4.0],
            vec![true, true, false, true],
            vec![2, 4],
        );
        let remasked = finish(
            &base_spec,
            vec![1.0, 2.0, 3.0, 4.0],
            vec![true, false, false, true],
            vec![2, 4],
        );
        let reboundaried = finish(
            &base_spec,
            vec![1.0, 2.0, 3.0, 4.0],
            vec![true, true, false, true],
            vec![1, 4],
        );
        let other_source = ActivationCacheSpec::new(
            7,
            base_spec.tensor_name(),
            4,
            1,
            ActivationDType::Float32,
            digest(11),
            2,
        )
        .expect("other source spec");
        let reprovenanced = finish(
            &other_source,
            vec![1.0, 2.0, 3.0, 4.0],
            vec![true, true, false, true],
            vec![2, 4],
        );

        for changed in [reordered, remasked, reboundaried, reprovenanced] {
            assert_ne!(changed, base);
        }
    }

    #[test]
    fn shards_have_canonical_indices_and_ranges() {
        let spec = spec(ActivationDType::BFloat16, 5, 1, 2);
        let mut builder = ActivationCacheBuilder::new(spec.clone());
        builder
            .ingest(chunk(
                &spec,
                0,
                5,
                vec![1.0, 2.0, 3.0, 4.0, 5.0],
                vec![true; 5],
                vec![3, 5],
            ))
            .expect("ingest");
        let cache = builder.finalize().expect("finalize");

        let observed: Vec<(u32, u64, u32, Vec<u64>)> = cache
            .shards()
            .iter()
            .map(|shard| {
                (
                    shard.index(),
                    shard.token_start(),
                    shard.token_count(),
                    shard.sequence_ends().to_vec(),
                )
            })
            .collect();
        assert_eq!(
            observed,
            vec![(0, 0, 2, vec![]), (1, 2, 2, vec![3]), (2, 4, 1, vec![5])]
        );
        assert!(
            cache
                .shards()
                .iter()
                .all(|shard| shard.digest() != digest(0))
        );
        for shard in cache.shards() {
            assert_eq!(
                u64::try_from(
                    cache
                        .encoded_shard(shard.index())
                        .expect("indexed canonical shard")
                        .len()
                )
                .expect("shard length fits u64"),
                shard.encoded_bytes()
            );
        }
    }

    #[test]
    fn byte_ledger_matches_the_canonical_encoding_exactly() {
        let spec = ActivationCacheSpec::new(1, "x", 3, 2, ActivationDType::Float32, digest(4), 2)
            .expect("spec");
        let mut builder = ActivationCacheBuilder::new(spec.clone());
        builder
            .ingest(chunk(
                &spec,
                0,
                3,
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                vec![true, false, true],
                vec![2, 3],
            ))
            .expect("ingest");
        let cache = builder.finalize().expect("finalize");
        let ledger = cache.byte_ledger();

        assert_eq!(ledger.manifest_bytes(), 73);
        assert_eq!(ledger.shard_header_bytes(), 64);
        assert_eq!(ledger.value_bytes(), 24);
        assert_eq!(ledger.mask_bytes(), 2);
        assert_eq!(ledger.boundary_bytes(), 16);
        assert_eq!(ledger.encoded_bytes(), 179);
        assert_eq!(ledger.cache_bytes(), 179);
        assert_eq!(cache.encoded().len(), 179);
    }

    #[test]
    fn rejects_gaps_duplicates_and_overlaps() {
        let spec = spec(ActivationDType::Float32, 4, 1, 2);
        let first = chunk(&spec, 0, 2, vec![1.0, 2.0], vec![true; 2], vec![2]);

        let mut duplicate = ActivationCacheBuilder::new(spec.clone());
        duplicate.ingest(first.clone()).expect("first ingest");
        assert!(matches!(
            duplicate.ingest(first),
            Err(ActivationCacheError::DuplicateRange { .. })
        ));

        let mut overlap = ActivationCacheBuilder::new(spec.clone());
        overlap
            .ingest(chunk(&spec, 0, 3, vec![1.0; 3], vec![true; 3], vec![3]))
            .expect("first overlap ingest");
        assert!(matches!(
            overlap.ingest(chunk(&spec, 2, 2, vec![1.0; 2], vec![true; 2], vec![4])),
            Err(ActivationCacheError::OverlappingRange { .. })
        ));

        let mut gap = ActivationCacheBuilder::new(spec.clone());
        gap.ingest(chunk(&spec, 2, 2, vec![1.0; 2], vec![true; 2], vec![4]))
            .expect("tail ingest");
        assert!(matches!(
            gap.finalize(),
            Err(ActivationCacheError::Gap {
                expected_start: 0,
                next_start: Some(2)
            })
        ));
    }

    #[test]
    fn rejects_malformed_masks_boundaries_values_and_ranges() {
        let spec = spec(ActivationDType::Float32, 4, 2, 2);

        assert!(matches!(
            ActivationChunk::new(&spec, 0, 2, vec![1.0; 4], vec![true], vec![2]),
            Err(ActivationCacheError::MaskLengthMismatch { .. })
        ));
        assert!(matches!(
            ActivationChunk::new(&spec, 0, 2, vec![1.0; 3], vec![true; 2], vec![2]),
            Err(ActivationCacheError::ValueCountMismatch { .. })
        ));
        assert!(matches!(
            ActivationChunk::new(
                &spec,
                0,
                2,
                vec![1.0, f32::NAN, 1.0, 1.0],
                vec![true; 2],
                vec![2]
            ),
            Err(ActivationCacheError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            ActivationChunk::new(&spec, 0, 2, vec![1.0; 4], vec![true; 2], vec![2, 2]),
            Err(ActivationCacheError::InvalidBoundary { .. })
        ));
        assert!(matches!(
            ActivationChunk::new(&spec, 3, 2, vec![1.0; 4], vec![true; 2], vec![4]),
            Err(ActivationCacheError::ChunkOutOfBounds { .. })
        ));
    }

    #[test]
    fn float16_encoding_rejects_finite_overflow() {
        let spec = spec(ActivationDType::Float16, 1, 1, 1);
        assert!(matches!(
            ActivationChunk::new(&spec, 0, 1, vec![70_000.0], vec![true], vec![1]),
            Err(ActivationCacheError::DTypeOverflow {
                dtype: ActivationDType::Float16,
                ..
            })
        ));
    }

    #[test]
    fn terminal_sequence_boundary_is_mandatory() {
        let spec = spec(ActivationDType::Float32, 2, 1, 2);
        let mut builder = ActivationCacheBuilder::new(spec.clone());
        builder
            .ingest(chunk(&spec, 0, 2, vec![1.0, 2.0], vec![true; 2], vec![]))
            .expect("ingest");
        assert!(matches!(
            builder.finalize(),
            Err(ActivationCacheError::MissingTerminalBoundary { total_tokens: 2 })
        ));
    }

    #[test]
    fn checked_arithmetic_rejects_impossible_tensor_shapes() {
        assert!(matches!(
            ActivationCacheSpec::new(
                0,
                "huge",
                u64::MAX,
                2,
                ActivationDType::Float32,
                digest(1),
                1
            ),
            Err(ActivationCacheError::ArithmeticOverflow { .. })
        ));
    }

    #[test]
    fn partial_builder_checkpoint_is_canonical_reopenable_and_resumable() {
        let spec = spec(ActivationDType::Float32, 6, 2, 2);
        let values: Vec<f32> = (0..12).map(|value| value as f32 / 8.0).collect();
        let mask = [true, false, true, true, false, true];

        let mut first_order = ActivationCacheBuilder::new(spec.clone());
        first_order
            .ingest(chunk(
                &spec,
                4,
                2,
                values[8..].to_vec(),
                mask[4..].to_vec(),
                vec![6],
            ))
            .expect("ingest tail");
        first_order
            .ingest(chunk(
                &spec,
                0,
                2,
                values[..4].to_vec(),
                mask[..2].to_vec(),
                vec![2],
            ))
            .expect("ingest head");

        let mut reverse_order = ActivationCacheBuilder::new(spec.clone());
        reverse_order
            .ingest(chunk(
                &spec,
                0,
                2,
                values[..4].to_vec(),
                mask[..2].to_vec(),
                vec![2],
            ))
            .expect("ingest head first");
        reverse_order
            .ingest(chunk(
                &spec,
                4,
                2,
                values[8..].to_vec(),
                mask[4..].to_vec(),
                vec![6],
            ))
            .expect("ingest tail second");

        let (checkpoint, checkpoint_digest) =
            first_order.encode_checkpoint().expect("encode checkpoint");
        let (reverse_checkpoint, reverse_digest) = reverse_order
            .encode_checkpoint()
            .expect("encode reverse checkpoint");
        assert_eq!(checkpoint, reverse_checkpoint);
        assert_eq!(checkpoint_digest, reverse_digest);

        let mut reopened =
            ActivationCacheBuilder::from_checkpoint_verified(&checkpoint, checkpoint_digest)
                .expect("verify checkpoint");
        assert_eq!(reopened.covered_tokens(), 4);
        let missing = reopened.missing_ranges();
        assert_eq!(missing.len(), 1);
        assert_eq!(missing[0], 2..4);
        let (reencoded, reencoded_digest) =
            reopened.encode_checkpoint().expect("re-encode checkpoint");
        assert_eq!(reencoded, checkpoint);
        assert_eq!(reencoded_digest, checkpoint_digest);

        reopened
            .ingest(chunk(
                &spec,
                2,
                2,
                values[4..8].to_vec(),
                mask[2..4].to_vec(),
                vec![4],
            ))
            .expect("resume missing middle");
        let resumed = reopened.finalize().expect("finalize resumed cache");

        let mut one_shot = ActivationCacheBuilder::new(spec.clone());
        one_shot
            .ingest(chunk(&spec, 0, 6, values, mask.to_vec(), vec![2, 4, 6]))
            .expect("ingest one shot");
        let one_shot = one_shot.finalize().expect("finalize one shot");
        assert_eq!(resumed.encoded(), one_shot.encoded());
        assert_eq!(resumed.digest(), one_shot.digest());
    }

    #[test]
    fn partial_checkpoint_reader_is_bounded_and_identity_verified() {
        let spec = spec(ActivationDType::Float16, 4, 1, 2);
        let mut builder = ActivationCacheBuilder::new(spec.clone());
        builder
            .ingest(chunk(
                &spec,
                2,
                2,
                vec![3.0, 4.0],
                vec![true, false],
                vec![4],
            ))
            .expect("ingest partial tail");
        let (checkpoint, checkpoint_digest) = builder.encode_checkpoint().expect("checkpoint");

        let reopened = ActivationCacheBuilder::read_checkpoint_from_verified(
            std::io::Cursor::new(&checkpoint),
            checkpoint.len() as u64,
            checkpoint_digest,
        )
        .expect("bounded checkpoint reopen");
        assert_eq!(
            reopened.encode_checkpoint().expect("re-encode").0,
            checkpoint
        );
        assert!(matches!(
            ActivationCacheBuilder::read_checkpoint_from_verified(
                std::io::Cursor::new(&checkpoint),
                checkpoint.len() as u64 - 1,
                checkpoint_digest,
            ),
            Err(ActivationCacheError::EncodedSizeLimitExceeded { .. })
        ));
        assert!(matches!(
            ActivationCacheBuilder::from_checkpoint_verified(&checkpoint, digest(44)),
            Err(ActivationCacheError::DigestMismatch { .. })
        ));
    }

    #[test]
    fn partial_checkpoint_rejects_corruption_truncation_trailing_and_noncanonical_order() {
        let spec = spec(ActivationDType::Float32, 4, 1, 2);
        let mut builder = ActivationCacheBuilder::new(spec.clone());
        builder
            .ingest(chunk(
                &spec,
                0,
                2,
                vec![1.0, 2.0],
                vec![true, false],
                vec![2],
            ))
            .expect("head");
        builder
            .ingest(chunk(
                &spec,
                2,
                2,
                vec![3.0, 4.0],
                vec![false, true],
                vec![4],
            ))
            .expect("tail");
        let (checkpoint, _) = builder.encode_checkpoint().expect("checkpoint");

        let mut corrupt = checkpoint.clone();
        corrupt[CHECKPOINT_MANIFEST_FIXED_BYTES as usize + spec.tensor_name().len() + 72] ^= 1;
        assert!(matches!(
            ActivationCacheBuilder::from_checkpoint(&corrupt),
            Err(ActivationCacheError::DigestMismatch { .. })
        ));
        assert!(
            ActivationCacheBuilder::from_checkpoint(&checkpoint[..checkpoint.len() - 1]).is_err()
        );
        let mut trailing = checkpoint.clone();
        trailing.push(0);
        assert!(ActivationCacheBuilder::from_checkpoint(&trailing).is_err());

        let manifest = CHECKPOINT_MANIFEST_FIXED_BYTES as usize + spec.tensor_name().len();
        let record_bytes = CHECKPOINT_CHUNK_HEADER_BYTES as usize + 8 + 1 + 8;
        let mut reordered = checkpoint.clone();
        let records_end = manifest + 2 * record_bytes;
        let (prefix, remainder) = reordered.split_at_mut(manifest);
        let _ = prefix;
        let (records, _) = remainder.split_at_mut(2 * record_bytes);
        let (first, second) = records.split_at_mut(record_bytes);
        first.swap_with_slice(second);
        let digest = hash_domain(CHECKPOINT_DOMAIN, &reordered[..records_end]);
        reordered[records_end..records_end + 32].copy_from_slice(digest.as_bytes());
        assert!(matches!(
            ActivationCacheBuilder::from_checkpoint(&reordered),
            Err(ActivationCacheError::NonCanonicalChunkOrder { .. })
        ));
    }

    #[test]
    fn failed_merge_is_atomic_and_successful_merge_preserves_canonical_order() {
        let spec = spec(ActivationDType::Float32, 6, 1, 2);
        let mut left = ActivationCacheBuilder::new(spec.clone());
        left.ingest(chunk(
            &spec,
            0,
            3,
            vec![1.0, 2.0, 3.0],
            vec![true; 3],
            vec![3],
        ))
        .expect("left");
        let before = left.encode_checkpoint().expect("before failed merge");
        let mut overlap = ActivationCacheBuilder::new(spec.clone());
        overlap
            .ingest(chunk(&spec, 2, 2, vec![3.0, 4.0], vec![true; 2], vec![4]))
            .expect("overlap");
        assert!(matches!(
            left.merge(overlap),
            Err(ActivationCacheError::OverlappingRange { .. })
        ));
        assert_eq!(
            left.encode_checkpoint().expect("after failed merge"),
            before
        );

        let mut right = ActivationCacheBuilder::new(spec.clone());
        right
            .ingest(chunk(
                &spec,
                3,
                3,
                vec![4.0, 5.0, 6.0],
                vec![true; 3],
                vec![6],
            ))
            .expect("right");
        left.merge(right).expect("non-overlapping merge");
        assert_eq!(left.covered_tokens(), 6);
        assert!(left.missing_ranges().is_empty());
        assert_eq!(
            left.finalize().expect("final cache").sequence_ends(),
            &[3, 6]
        );
    }

    #[test]
    fn bounded_final_reader_can_verify_expected_identity_without_a_second_copy() {
        let original = finalized_cache();
        let reopened = ActivationCache::read_from_verified(
            std::io::Cursor::new(original.encoded()),
            original.byte_ledger().encoded_bytes(),
            original.digest(),
        )
        .expect("verified bounded reopen");
        assert_eq!(reopened, original);
        assert!(matches!(
            ActivationCache::read_from_verified(
                std::io::Cursor::new(original.encoded()),
                original.byte_ledger().encoded_bytes(),
                digest(55),
            ),
            Err(ActivationCacheError::DigestMismatch { .. })
        ));
    }
}

use half::{bf16, f16};
use std::fmt;
use std::io::Read;
use std::ops::Range;

const CACHE_MAGIC: [u8; 4] = *b"S2AC";
const CACHE_VERSION: u16 = 1;
const CACHE_MANIFEST_FIXED_BYTES: u64 = 72;
const SHARD_HEADER_BYTES: u64 = 32;
const SCHEMA_DOMAIN: &[u8] = b"tritium salt v2 activation schema v1\0";
const CHUNK_DOMAIN: &[u8] = b"tritium salt v2 activation chunk v1\0";
const SHARD_DOMAIN: &[u8] = b"tritium salt v2 activation shard v1\0";
const CACHE_DOMAIN: &[u8] = b"tritium salt v2 activation cache v1\0";
const CHECKPOINT_MAGIC: [u8; 4] = *b"S2CP";
const CHECKPOINT_VERSION: u16 = 1;
const CHECKPOINT_MANIFEST_FIXED_BYTES: u64 = 84;
const CHECKPOINT_CHUNK_HEADER_BYTES: u64 = 72;
const CHECKPOINT_DIGEST_BYTES: u64 = 32;
const CHECKPOINT_DOMAIN: &[u8] = b"tritium salt v2 activation checkpoint v1\0";

/// A BLAKE3 content identifier used by SALT V2 activation artifacts.
#[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ActivationDigest([u8; 32]);

impl ActivationDigest {
    /// Constructs a digest from its exact 32-byte representation.
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Borrows the exact 32-byte representation.
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Returns the exact 32-byte representation.
    pub const fn into_bytes(self) -> [u8; 32] {
        self.0
    }
}

impl fmt::Display for ActivationDigest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for ActivationDigest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "ActivationDigest({self})")
    }
}

/// Canonical scalar encoding used by an activation cache.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ActivationDType {
    /// IEEE-754 binary32, little endian.
    Float32,
    /// IEEE-754 binary16, little endian.
    Float16,
    /// Brain floating point binary16, little endian.
    BFloat16,
}

impl ActivationDType {
    /// Returns the number of encoded bytes per scalar.
    pub const fn encoded_width(self) -> u8 {
        match self {
            Self::Float32 => 4,
            Self::Float16 | Self::BFloat16 => 2,
        }
    }

    const fn format_code(self) -> u8 {
        match self {
            Self::Float32 => 1,
            Self::Float16 => 2,
            Self::BFloat16 => 3,
        }
    }
}

/// Failure while defining, assembling, or encoding a SALT V2 activation cache.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ActivationCacheError {
    /// The canonical cache magic did not match `S2AC`.
    InvalidCacheMagic {
        /// Four bytes found at the start of the input.
        got: [u8; 4],
    },
    /// The canonical partial-builder checkpoint magic did not match `S2CP`.
    InvalidCheckpointMagic {
        /// Four bytes found at the start of the checkpoint.
        got: [u8; 4],
    },
    /// The cache format version is not supported by this decoder.
    UnsupportedCacheVersion {
        /// Unsupported little-endian version.
        got: u16,
    },
    /// The partial-builder checkpoint version is not supported by this decoder.
    UnsupportedCheckpointVersion {
        /// Unsupported little-endian version.
        got: u16,
    },
    /// The manifest reserved byte was nonzero.
    NonZeroReservedByte {
        /// Noncanonical reserved value.
        got: u8,
    },
    /// The manifest dtype code was not one of the canonical values 1, 2, or 3.
    InvalidDTypeCode {
        /// Unsupported dtype code.
        got: u8,
    },
    /// Canonical bytes ended before a declared field or payload was complete.
    TruncatedEncoding {
        /// Field or payload being decoded.
        context: &'static str,
        /// Bytes required for this read.
        needed: u64,
        /// Bytes still available.
        remaining: u64,
    },
    /// Bytes remained after the final canonical shard record.
    TrailingBytes {
        /// Number of unexpected terminal bytes.
        count: u64,
    },
    /// The persisted tensor name was not valid UTF-8.
    InvalidTensorNameUtf8,
    /// The persisted shard count differed from the schema-derived canonical count.
    ShardCountMismatch {
        /// Schema-derived shard count.
        expected: u32,
        /// Persisted shard count.
        got: u32,
    },
    /// A shard record did not carry its contiguous canonical index.
    ShardIndexMismatch {
        /// Required shard index.
        expected: u32,
        /// Persisted shard index.
        got: u32,
    },
    /// A shard's token interval differed from its schema-derived fixed range.
    ShardRangeMismatch {
        /// Shard index being decoded.
        index: u32,
        /// Required global token start.
        expected_start: u64,
        /// Persisted global token start.
        got_start: u64,
        /// Required token count.
        expected_count: u32,
        /// Persisted token count.
        got_count: u32,
    },
    /// A shard component length differed from its shape-derived canonical length.
    ShardLengthMismatch {
        /// Shard index being decoded.
        index: u32,
        /// Component whose length was invalid.
        component: &'static str,
        /// Required byte count.
        expected: u64,
        /// Persisted byte count.
        got: u64,
    },
    /// A bit-packed mask set unused high bits in its terminal byte.
    NonCanonicalMask {
        /// Shard containing the noncanonical mask.
        index: u32,
    },
    /// A persisted scalar encoding represented NaN or infinity.
    NonFiniteEncodedValue {
        /// Global row-major scalar index.
        index: u64,
    },
    /// Recomputed complete-cache identity differed from the caller's expected identity.
    DigestMismatch {
        /// Required content digest.
        expected: ActivationDigest,
        /// Digest recomputed from canonical bytes.
        got: ActivationDigest,
    },
    /// A checkpoint chunk's persisted content digest was not canonical.
    CheckpointChunkDigestMismatch {
        /// Inclusive global token start of the affected chunk.
        token_start: u64,
        /// Persisted chunk digest.
        expected: ActivationDigest,
        /// Digest recomputed from schema, range, values, mask, and boundaries.
        got: ActivationDigest,
    },
    /// A reader produced more bytes than the caller's explicit durability bound.
    EncodedSizeLimitExceeded {
        /// Maximum accepted byte count.
        limit: u64,
    },
    /// Reading persisted cache bytes failed.
    ReadFailed {
        /// Portable I/O failure category.
        kind: std::io::ErrorKind,
    },
    /// The tensor name was empty, whitespace-only, or contained a NUL byte.
    InvalidTensorName,
    /// A source digest of all zeroes was supplied, indicating missing provenance.
    MissingSourceDigest,
    /// A required tensor or shard dimension was zero.
    ZeroDimension {
        /// Name of the invalid dimension.
        field: &'static str,
    },
    /// Checked arithmetic overflowed while deriving a shape or byte count.
    ArithmeticOverflow {
        /// Operation whose result was not representable.
        context: &'static str,
    },
    /// The canonical shard count did not fit the format's `u32` index space.
    TooManyShards {
        /// Number of shards required by the schema.
        count: u64,
    },
    /// A chunk declared zero tokens.
    EmptyChunk,
    /// A chunk's token range was outside the schema's tensor shape.
    ChunkOutOfBounds {
        /// Inclusive first token offset.
        start: u64,
        /// Number of tokens in the chunk.
        count: u64,
        /// Total token count declared by the schema.
        total: u64,
    },
    /// A chunk did not carry exactly one mask entry per declared token.
    MaskLengthMismatch {
        /// Required number of mask entries.
        expected: u64,
        /// Supplied number of mask entries.
        got: u64,
    },
    /// A chunk's row-major value count did not match its token and feature dimensions.
    ValueCountMismatch {
        /// Required number of scalar values.
        expected: u64,
        /// Supplied number of scalar values.
        got: u64,
    },
    /// A source activation was NaN or infinite.
    NonFiniteValue {
        /// Row-major scalar index containing the invalid value.
        index: u64,
    },
    /// A finite source activation overflowed the requested storage dtype.
    DTypeOverflow {
        /// Row-major scalar index containing the overflowing value.
        index: u64,
        /// Requested canonical storage dtype.
        dtype: ActivationDType,
    },
    /// Sequence ends were not strictly increasing within the chunk's token interval.
    InvalidBoundary {
        /// Previous exclusive sequence end, or the chunk start for the first entry.
        previous: u64,
        /// Invalid exclusive sequence end.
        boundary: u64,
        /// Inclusive first token offset of the chunk.
        chunk_start: u64,
        /// Exclusive last token offset of the chunk.
        chunk_end: u64,
    },
    /// A chunk was created for a different layer, tensor, shape, dtype, source, or shard policy.
    SchemaMismatch {
        /// Schema digest required by the destination builder.
        expected: ActivationDigest,
        /// Schema digest carried by the chunk or merged builder.
        got: ActivationDigest,
    },
    /// The exact token range had already been ingested.
    DuplicateRange {
        /// Inclusive first token offset of the duplicate range.
        start: u64,
        /// Exclusive last token offset of the duplicate range.
        end: u64,
    },
    /// A token range intersected a range already present in the builder.
    OverlappingRange {
        /// Inclusive first token offset of the incoming range.
        start: u64,
        /// Exclusive last token offset of the incoming range.
        end: u64,
        /// Inclusive first token offset of the conflicting range.
        existing_start: u64,
        /// Exclusive last token offset of the conflicting range.
        existing_end: u64,
    },
    /// Checkpoint chunks were not serialized in strictly increasing range order.
    NonCanonicalChunkOrder {
        /// Start of the previous checkpoint chunk.
        previous_start: u64,
        /// Start of the noncanonical next chunk.
        start: u64,
    },
    /// A checkpoint's declared covered-token count did not match its chunks.
    CoveredTokenMismatch {
        /// Sum of validated chunk token counts.
        expected: u64,
        /// Persisted covered-token count.
        got: u64,
    },
    /// Finalization found a gap in the zero-based token stream.
    Gap {
        /// Token offset at which data was required.
        expected_start: u64,
        /// Start of the next available range, or `None` for a trailing gap.
        next_start: Option<u64>,
    },
    /// The global exclusive sequence-end list did not end at the tensor's token count.
    MissingTerminalBoundary {
        /// Required terminal sequence end.
        total_tokens: u64,
    },
    /// The same global exclusive sequence end appeared more than once.
    DuplicateBoundary {
        /// Repeated exclusive sequence end.
        boundary: u64,
    },
    /// Allocation of the canonical artifact backing store failed.
    AllocationFailed {
        /// Number of bytes requested for the backing store.
        bytes: u64,
    },
    /// Internal canonical byte accounting differed from the bytes actually emitted.
    AccountingMismatch {
        /// Byte count computed before encoding.
        expected: u64,
        /// Byte count emitted by the encoder.
        got: u64,
    },
}

impl fmt::Display for ActivationCacheError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCacheMagic { got } => {
                write!(formatter, "invalid activation cache magic {got:02x?}")
            }
            Self::InvalidCheckpointMagic { got } => {
                write!(formatter, "invalid activation checkpoint magic {got:02x?}")
            }
            Self::UnsupportedCacheVersion { got } => {
                write!(formatter, "unsupported activation cache version {got}")
            }
            Self::UnsupportedCheckpointVersion { got } => {
                write!(formatter, "unsupported activation checkpoint version {got}")
            }
            Self::NonZeroReservedByte { got } => {
                write!(
                    formatter,
                    "activation cache reserved byte is {got}, expected zero"
                )
            }
            Self::InvalidDTypeCode { got } => {
                write!(formatter, "invalid activation cache dtype code {got}")
            }
            Self::TruncatedEncoding {
                context,
                needed,
                remaining,
            } => write!(
                formatter,
                "activation cache is truncated while reading {context}: need {needed} bytes, have {remaining}"
            ),
            Self::TrailingBytes { count } => {
                write!(formatter, "activation cache has {count} trailing bytes")
            }
            Self::InvalidTensorNameUtf8 => {
                write!(formatter, "activation cache tensor name is not valid UTF-8")
            }
            Self::ShardCountMismatch { expected, got } => write!(
                formatter,
                "activation shard count mismatch: expected {expected}, got {got}"
            ),
            Self::ShardIndexMismatch { expected, got } => write!(
                formatter,
                "activation shard index mismatch: expected {expected}, got {got}"
            ),
            Self::ShardRangeMismatch {
                index,
                expected_start,
                got_start,
                expected_count,
                got_count,
            } => write!(
                formatter,
                "activation shard {index} range mismatch: expected [{expected_start}, +{expected_count}), got [{got_start}, +{got_count})"
            ),
            Self::ShardLengthMismatch {
                index,
                component,
                expected,
                got,
            } => write!(
                formatter,
                "activation shard {index} {component} length mismatch: expected {expected}, got {got}"
            ),
            Self::NonCanonicalMask { index } => {
                write!(
                    formatter,
                    "activation shard {index} mask has nonzero padding bits"
                )
            }
            Self::NonFiniteEncodedValue { index } => {
                write!(formatter, "encoded activation value {index} is not finite")
            }
            Self::DigestMismatch { expected, got } => write!(
                formatter,
                "activation artifact digest mismatch: expected {expected}, got {got}"
            ),
            Self::CheckpointChunkDigestMismatch {
                token_start,
                expected,
                got,
            } => write!(
                formatter,
                "activation checkpoint chunk at {token_start} has digest {expected}, recomputed {got}"
            ),
            Self::EncodedSizeLimitExceeded { limit } => write!(
                formatter,
                "activation cache exceeds the reader limit of {limit} bytes"
            ),
            Self::ReadFailed { kind } => {
                write!(formatter, "failed to read activation cache bytes: {kind:?}")
            }
            Self::InvalidTensorName => write!(formatter, "activation tensor name is invalid"),
            Self::MissingSourceDigest => {
                write!(formatter, "activation source digest must be present")
            }
            Self::ZeroDimension { field } => {
                write!(formatter, "activation {field} must be greater than zero")
            }
            Self::ArithmeticOverflow { context } => {
                write!(
                    formatter,
                    "activation arithmetic overflow while computing {context}"
                )
            }
            Self::TooManyShards { count } => {
                write!(
                    formatter,
                    "activation cache requires {count} shards, exceeding u32"
                )
            }
            Self::EmptyChunk => write!(formatter, "activation chunk must contain tokens"),
            Self::ChunkOutOfBounds {
                start,
                count,
                total,
            } => write!(
                formatter,
                "activation chunk [{start}, {start}+{count}) is outside 0..{total}"
            ),
            Self::MaskLengthMismatch { expected, got } => write!(
                formatter,
                "activation token-mask length mismatch: expected {expected}, got {got}"
            ),
            Self::ValueCountMismatch { expected, got } => write!(
                formatter,
                "activation value count mismatch: expected {expected}, got {got}"
            ),
            Self::NonFiniteValue { index } => {
                write!(formatter, "activation value {index} is not finite")
            }
            Self::DTypeOverflow { index, dtype } => write!(
                formatter,
                "activation value {index} overflows storage dtype {dtype:?}"
            ),
            Self::InvalidBoundary {
                previous,
                boundary,
                chunk_start,
                chunk_end,
            } => write!(
                formatter,
                "activation sequence end {boundary} after {previous} is invalid for chunk [{chunk_start}, {chunk_end})"
            ),
            Self::SchemaMismatch { expected, got } => write!(
                formatter,
                "activation schema mismatch: expected {expected}, got {got}"
            ),
            Self::DuplicateRange { start, end } => {
                write!(formatter, "activation range [{start}, {end}) is duplicated")
            }
            Self::OverlappingRange {
                start,
                end,
                existing_start,
                existing_end,
            } => write!(
                formatter,
                "activation range [{start}, {end}) overlaps [{existing_start}, {existing_end})"
            ),
            Self::NonCanonicalChunkOrder {
                previous_start,
                start,
            } => write!(
                formatter,
                "activation checkpoint chunk start {start} does not follow {previous_start} canonically"
            ),
            Self::CoveredTokenMismatch { expected, got } => write!(
                formatter,
                "activation checkpoint covers {expected} tokens but declares {got}"
            ),
            Self::Gap {
                expected_start,
                next_start,
            } => match next_start {
                Some(next) => write!(
                    formatter,
                    "activation cache has a gap at {expected_start} before range {next}"
                ),
                None => write!(
                    formatter,
                    "activation cache has a trailing gap at {expected_start}"
                ),
            },
            Self::MissingTerminalBoundary { total_tokens } => write!(
                formatter,
                "activation sequence ends must terminate at token {total_tokens}"
            ),
            Self::DuplicateBoundary { boundary } => write!(
                formatter,
                "activation sequence end {boundary} is duplicated"
            ),
            Self::AllocationFailed { bytes } => write!(
                formatter,
                "could not allocate {bytes} bytes for activation cache"
            ),
            Self::AccountingMismatch { expected, got } => write!(
                formatter,
                "activation byte accounting mismatch: expected {expected}, emitted {got}"
            ),
        }
    }
}

impl std::error::Error for ActivationCacheError {}

/// Immutable schema and provenance for one layer tensor's activation cache.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActivationCacheSpec {
    layer_index: u32,
    tensor_name: String,
    total_tokens: u64,
    feature_width: u64,
    dtype: ActivationDType,
    source_digest: ActivationDigest,
    shard_tokens: u32,
    shard_count: u32,
    schema_digest: ActivationDigest,
}

impl ActivationCacheSpec {
    /// Validates a layer tensor shape, its source provenance, and canonical shard policy.
    pub fn new(
        layer_index: u32,
        tensor_name: impl Into<String>,
        total_tokens: u64,
        feature_width: u64,
        dtype: ActivationDType,
        source_digest: ActivationDigest,
        shard_tokens: u32,
    ) -> Result<Self, ActivationCacheError> {
        let tensor_name = tensor_name.into();
        if tensor_name.trim().is_empty() || tensor_name.as_bytes().contains(&0) {
            return Err(ActivationCacheError::InvalidTensorName);
        }
        let _ = u32::try_from(tensor_name.len()).map_err(|_| {
            ActivationCacheError::ArithmeticOverflow {
                context: "tensor-name length",
            }
        })?;
        if total_tokens == 0 {
            return Err(ActivationCacheError::ZeroDimension {
                field: "token count",
            });
        }
        if feature_width == 0 {
            return Err(ActivationCacheError::ZeroDimension {
                field: "feature width",
            });
        }
        if shard_tokens == 0 {
            return Err(ActivationCacheError::ZeroDimension {
                field: "shard token count",
            });
        }
        if source_digest.as_bytes().iter().all(|byte| *byte == 0) {
            return Err(ActivationCacheError::MissingSourceDigest);
        }

        let scalar_count = checked_mul(total_tokens, feature_width, "tensor element count")?;
        checked_mul(
            scalar_count,
            u64::from(dtype.encoded_width()),
            "encoded tensor bytes",
        )?;
        let shard_count = total_tokens
            .checked_sub(1)
            .and_then(|value| value.checked_div(u64::from(shard_tokens)))
            .and_then(|value| value.checked_add(1))
            .ok_or(ActivationCacheError::ArithmeticOverflow {
                context: "canonical shard count",
            })?;
        let shard_count = u32::try_from(shard_count)
            .map_err(|_| ActivationCacheError::TooManyShards { count: shard_count })?;
        let schema_digest = compute_schema_digest(
            layer_index,
            &tensor_name,
            total_tokens,
            feature_width,
            dtype,
            source_digest,
            shard_tokens,
        );

        Ok(Self {
            layer_index,
            tensor_name,
            total_tokens,
            feature_width,
            dtype,
            source_digest,
            shard_tokens,
            shard_count,
            schema_digest,
        })
    }

    /// Returns the zero-based transformer-layer index.
    pub const fn layer_index(&self) -> u32 {
        self.layer_index
    }

    /// Returns the architecture tensor name.
    pub fn tensor_name(&self) -> &str {
        &self.tensor_name
    }

    /// Returns the number of flattened token rows in the tensor.
    pub const fn total_tokens(&self) -> u64 {
        self.total_tokens
    }

    /// Returns the scalar feature count in each token row.
    pub const fn feature_width(&self) -> u64 {
        self.feature_width
    }

    /// Returns the canonical scalar storage dtype.
    pub const fn dtype(&self) -> ActivationDType {
        self.dtype
    }

    /// Returns the source-provenance envelope digest.
    ///
    /// Callers should bind the model, tokenizer, dataset revisions, sampling seed, and ordered
    /// token stream into this envelope before constructing the cache.
    pub const fn source_digest(&self) -> ActivationDigest {
        self.source_digest
    }

    /// Returns the maximum token count in each canonical shard.
    pub const fn shard_tokens(&self) -> u32 {
        self.shard_tokens
    }

    /// Returns the exact number of canonical shards.
    pub const fn shard_count(&self) -> u32 {
        self.shard_count
    }

    /// Returns the digest binding layer, tensor, shape, dtype, source, and shard policy.
    pub const fn schema_digest(&self) -> ActivationDigest {
        self.schema_digest
    }
}

/// A validated, dtype-encoded token interval that can be checkpointed and merged on resume.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActivationChunk {
    schema_digest: ActivationDigest,
    token_start: u64,
    token_count: u64,
    encoded_values: Vec<u8>,
    token_mask: Vec<bool>,
    sequence_ends: Vec<u64>,
    digest: ActivationDigest,
}

impl ActivationChunk {
    /// Validates and canonically encodes a row-major activation interval.
    ///
    /// Sequence ends are global exclusive token offsets. Every supplied end must be in
    /// `(token_start, token_start + token_count]`; a sequence spanning chunks is recorded by the
    /// chunk containing its end.
    pub fn new(
        spec: &ActivationCacheSpec,
        token_start: u64,
        token_count: u64,
        values: Vec<f32>,
        token_mask: Vec<bool>,
        sequence_ends: Vec<u64>,
    ) -> Result<Self, ActivationCacheError> {
        if token_count == 0 {
            return Err(ActivationCacheError::EmptyChunk);
        }
        let token_end = token_start.checked_add(token_count).ok_or(
            ActivationCacheError::ArithmeticOverflow {
                context: "chunk token end",
            },
        )?;
        if token_start >= spec.total_tokens || token_end > spec.total_tokens {
            return Err(ActivationCacheError::ChunkOutOfBounds {
                start: token_start,
                count: token_count,
                total: spec.total_tokens,
            });
        }

        let mask_count = len_u64(token_mask.len(), "token-mask length")?;
        if mask_count != token_count {
            return Err(ActivationCacheError::MaskLengthMismatch {
                expected: token_count,
                got: mask_count,
            });
        }
        let expected_values =
            checked_mul(token_count, spec.feature_width, "chunk scalar value count")?;
        let got_values = len_u64(values.len(), "chunk scalar value count")?;
        if got_values != expected_values {
            return Err(ActivationCacheError::ValueCountMismatch {
                expected: expected_values,
                got: got_values,
            });
        }

        let mut previous = token_start;
        for &boundary in &sequence_ends {
            if boundary <= previous || boundary > token_end {
                return Err(ActivationCacheError::InvalidBoundary {
                    previous,
                    boundary,
                    chunk_start: token_start,
                    chunk_end: token_end,
                });
            }
            previous = boundary;
        }

        let encoded_len = checked_mul(
            expected_values,
            u64::from(spec.dtype.encoded_width()),
            "encoded chunk bytes",
        )?;
        let encoded_capacity = usize_from_u64(encoded_len, "encoded chunk allocation")?;
        let mut encoded_values = Vec::new();
        encoded_values
            .try_reserve_exact(encoded_capacity)
            .map_err(|_| ActivationCacheError::AllocationFailed { bytes: encoded_len })?;
        for (index, value) in values.into_iter().enumerate() {
            let index = len_u64(index, "activation value index")?;
            encode_value(spec.dtype, value, index, &mut encoded_values)?;
        }

        let digest = compute_chunk_digest(
            spec.schema_digest,
            token_start,
            token_count,
            &encoded_values,
            &token_mask,
            &sequence_ends,
        )?;
        Ok(Self {
            schema_digest: spec.schema_digest,
            token_start,
            token_count,
            encoded_values,
            token_mask,
            sequence_ends,
            digest,
        })
    }

    /// Returns the schema digest this chunk was encoded against.
    pub const fn schema_digest(&self) -> ActivationDigest {
        self.schema_digest
    }

    /// Returns the inclusive first global token offset.
    pub const fn token_start(&self) -> u64 {
        self.token_start
    }

    /// Returns the number of token rows in this interval.
    pub const fn token_count(&self) -> u64 {
        self.token_count
    }

    /// Returns the exclusive final global token offset.
    pub const fn token_end(&self) -> u64 {
        self.token_start + self.token_count
    }

    /// Returns canonical little-endian scalar bytes for checkpoint persistence.
    pub fn encoded_values(&self) -> &[u8] {
        &self.encoded_values
    }

    /// Returns one validity bit per token row.
    pub fn token_mask(&self) -> &[bool] {
        &self.token_mask
    }

    /// Returns global exclusive sequence ends carried by this interval.
    pub fn sequence_ends(&self) -> &[u64] {
        &self.sequence_ends
    }

    /// Returns the chunk's content digest, including schema, range, values, mask, and boundaries.
    pub const fn digest(&self) -> ActivationDigest {
        self.digest
    }
}

/// Resume-friendly assembler for validated activation chunks.
#[derive(Clone, Debug)]
pub struct ActivationCacheBuilder {
    spec: ActivationCacheSpec,
    chunks: Vec<ActivationChunk>,
    covered_tokens: u64,
}

impl ActivationCacheBuilder {
    /// Creates an empty builder for a validated cache schema.
    pub fn new(spec: ActivationCacheSpec) -> Self {
        Self {
            spec,
            chunks: Vec::new(),
            covered_tokens: 0,
        }
    }

    /// Returns the destination schema.
    pub const fn spec(&self) -> &ActivationCacheSpec {
        &self.spec
    }

    /// Returns the number of non-overlapping token rows currently present.
    pub const fn covered_tokens(&self) -> u64 {
        self.covered_tokens
    }

    /// Returns missing global token intervals in ascending order.
    pub fn missing_ranges(&self) -> Vec<Range<u64>> {
        let mut missing = Vec::new();
        let mut expected = 0;
        for chunk in &self.chunks {
            if chunk.token_start > expected {
                missing.push(expected..chunk.token_start);
            }
            expected = chunk.token_end();
        }
        if expected < self.spec.total_tokens {
            missing.push(expected..self.spec.total_tokens);
        }
        missing
    }

    /// Atomically adds a non-overlapping chunk created from the same exact schema.
    pub fn ingest(&mut self, chunk: ActivationChunk) -> Result<(), ActivationCacheError> {
        if chunk.schema_digest != self.spec.schema_digest {
            return Err(ActivationCacheError::SchemaMismatch {
                expected: self.spec.schema_digest,
                got: chunk.schema_digest,
            });
        }
        let start = chunk.token_start;
        let end = chunk.token_end();
        let insertion = match self
            .chunks
            .binary_search_by_key(&start, |existing| existing.token_start)
        {
            Ok(index) => {
                let existing = &self.chunks[index];
                if existing.token_end() == end {
                    return Err(ActivationCacheError::DuplicateRange { start, end });
                }
                return Err(ActivationCacheError::OverlappingRange {
                    start,
                    end,
                    existing_start: existing.token_start,
                    existing_end: existing.token_end(),
                });
            }
            Err(index) => index,
        };
        if let Some(previous) = insertion
            .checked_sub(1)
            .and_then(|index| self.chunks.get(index))
            && previous.token_end() > start
        {
            return Err(ActivationCacheError::OverlappingRange {
                start,
                end,
                existing_start: previous.token_start,
                existing_end: previous.token_end(),
            });
        }
        if let Some(next) = self.chunks.get(insertion)
            && next.token_start < end
        {
            return Err(ActivationCacheError::OverlappingRange {
                start,
                end,
                existing_start: next.token_start,
                existing_end: next.token_end(),
            });
        }

        let covered_tokens = self.covered_tokens.checked_add(chunk.token_count).ok_or(
            ActivationCacheError::ArithmeticOverflow {
                context: "covered token count",
            },
        )?;
        let chunk_allocation =
            u64::try_from(std::mem::size_of::<ActivationChunk>()).map_err(|_| {
                ActivationCacheError::ArithmeticOverflow {
                    context: "chunk allocation size",
                }
            })?;
        self.chunks
            .try_reserve(1)
            .map_err(|_| ActivationCacheError::AllocationFailed {
                bytes: chunk_allocation,
            })?;
        self.chunks.insert(insertion, chunk);
        self.covered_tokens = covered_tokens;
        Ok(())
    }

    /// Atomically merges a checkpointed partial builder, independent of its ingestion order.
    pub fn merge(&mut self, other: Self) -> Result<(), ActivationCacheError> {
        if self.spec != other.spec {
            return Err(ActivationCacheError::SchemaMismatch {
                expected: self.spec.schema_digest,
                got: other.spec.schema_digest,
            });
        }

        // Validate every cross-builder relationship before changing logical state. Capacity may
        // grow only after all fallible semantic checks have passed; a failed reservation leaves
        // the existing vector and covered-token count unchanged.
        for incoming in &other.chunks {
            let start = incoming.token_start;
            let end = incoming.token_end();
            let insertion = match self
                .chunks
                .binary_search_by_key(&start, |existing| existing.token_start)
            {
                Ok(index) => {
                    let existing = &self.chunks[index];
                    if existing.token_end() == end {
                        return Err(ActivationCacheError::DuplicateRange { start, end });
                    }
                    return Err(ActivationCacheError::OverlappingRange {
                        start,
                        end,
                        existing_start: existing.token_start,
                        existing_end: existing.token_end(),
                    });
                }
                Err(index) => index,
            };
            if let Some(previous) = insertion
                .checked_sub(1)
                .and_then(|index| self.chunks.get(index))
                && previous.token_end() > start
            {
                return Err(ActivationCacheError::OverlappingRange {
                    start,
                    end,
                    existing_start: previous.token_start,
                    existing_end: previous.token_end(),
                });
            }
            if let Some(next) = self.chunks.get(insertion)
                && next.token_start < end
            {
                return Err(ActivationCacheError::OverlappingRange {
                    start,
                    end,
                    existing_start: next.token_start,
                    existing_end: next.token_end(),
                });
            }
        }
        let covered_tokens = self
            .covered_tokens
            .checked_add(other.covered_tokens)
            .ok_or(ActivationCacheError::ArithmeticOverflow {
                context: "merged covered token count",
            })?;
        let merged_allocation = len_u64(other.chunks.len(), "merged chunk allocation")?;
        self.chunks.try_reserve(other.chunks.len()).map_err(|_| {
            ActivationCacheError::AllocationFailed {
                bytes: merged_allocation,
            }
        })?;
        self.chunks.extend(other.chunks);
        self.chunks.sort_unstable_by_key(|chunk| chunk.token_start);
        self.covered_tokens = covered_tokens;
        Ok(())
    }

    /// Canonically encodes this partial builder for durable crash recovery.
    ///
    /// Chunks are emitted in range order regardless of ingestion order. Returned digest covers
    /// every byte before the terminal digest field and is stored in that field as well.
    pub fn encode_checkpoint(&self) -> Result<(Vec<u8>, ActivationDigest), ActivationCacheError> {
        encode_builder_checkpoint(self)
    }

    /// Reopens and fully validates a canonical partial-builder checkpoint.
    pub fn from_checkpoint(encoded: &[u8]) -> Result<Self, ActivationCacheError> {
        decode_builder_checkpoint(encoded, None)
    }

    /// Reopens a partial checkpoint and requires its recomputed identity to match `expected`.
    pub fn from_checkpoint_verified(
        encoded: &[u8],
        expected: ActivationDigest,
    ) -> Result<Self, ActivationCacheError> {
        decode_builder_checkpoint(encoded, Some(expected))
    }

    /// Reads a bounded partial checkpoint, validates its stored and expected identities, and
    /// returns a resume-ready builder.
    pub fn read_checkpoint_from_verified<R: Read>(
        reader: R,
        max_encoded_bytes: u64,
        expected: ActivationDigest,
    ) -> Result<Self, ActivationCacheError> {
        let encoded = read_bounded(reader, max_encoded_bytes)?;
        Self::from_checkpoint_verified(&encoded, expected)
    }

    /// Canonicalizes complete chunks into fixed shards and a content-addressed artifact.
    pub fn finalize(self) -> Result<ActivationCache, ActivationCacheError> {
        let mut expected = 0;
        let mut sequence_ends = Vec::new();
        for chunk in &self.chunks {
            if chunk.token_start != expected {
                return Err(ActivationCacheError::Gap {
                    expected_start: expected,
                    next_start: Some(chunk.token_start),
                });
            }
            expected = chunk.token_end();
            let boundary_allocation =
                len_u64(chunk.sequence_ends.len(), "sequence boundary allocation")?;
            sequence_ends
                .try_reserve(chunk.sequence_ends.len())
                .map_err(|_| ActivationCacheError::AllocationFailed {
                    bytes: boundary_allocation,
                })?;
            for &boundary in &chunk.sequence_ends {
                if sequence_ends.last().copied() == Some(boundary) {
                    return Err(ActivationCacheError::DuplicateBoundary { boundary });
                }
                sequence_ends.push(boundary);
            }
        }
        if expected != self.spec.total_tokens {
            return Err(ActivationCacheError::Gap {
                expected_start: expected,
                next_start: None,
            });
        }
        if sequence_ends.last().copied() != Some(self.spec.total_tokens) {
            return Err(ActivationCacheError::MissingTerminalBoundary {
                total_tokens: self.spec.total_tokens,
            });
        }

        let value_bytes = checked_mul(
            checked_mul(
                self.spec.total_tokens,
                self.spec.feature_width,
                "cache scalar count",
            )?,
            u64::from(self.spec.dtype.encoded_width()),
            "cache value bytes",
        )?;
        let value_capacity = usize_from_u64(value_bytes, "cache value allocation")?;
        let token_capacity = usize_from_u64(self.spec.total_tokens, "cache mask allocation")?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(value_capacity)
            .map_err(|_| ActivationCacheError::AllocationFailed { bytes: value_bytes })?;
        let mut token_mask = Vec::new();
        token_mask.try_reserve_exact(token_capacity).map_err(|_| {
            ActivationCacheError::AllocationFailed {
                bytes: self.spec.total_tokens,
            }
        })?;
        for chunk in self.chunks {
            values.extend_from_slice(&chunk.encoded_values);
            token_mask.extend_from_slice(&chunk.token_mask);
        }
        encode_cache(self.spec, values, token_mask, sequence_ends)
    }
}

fn encode_builder_checkpoint(
    builder: &ActivationCacheBuilder,
) -> Result<(Vec<u8>, ActivationDigest), ActivationCacheError> {
    let name_bytes = len_u64(
        builder.spec.tensor_name.len(),
        "checkpoint tensor-name bytes",
    )?;
    let mut encoded_bytes = checked_add(
        CHECKPOINT_MANIFEST_FIXED_BYTES,
        name_bytes,
        "checkpoint manifest bytes",
    )?;
    for chunk in &builder.chunks {
        let packed_mask_bytes = chunk.token_count.div_ceil(8);
        let boundary_bytes = checked_mul(
            len_u64(chunk.sequence_ends.len(), "checkpoint boundary count")?,
            8,
            "checkpoint boundary bytes",
        )?;
        let payload_bytes = checked_add(
            checked_add(
                len_u64(chunk.encoded_values.len(), "checkpoint value bytes")?,
                packed_mask_bytes,
                "checkpoint values and mask",
            )?,
            boundary_bytes,
            "checkpoint chunk payload",
        )?;
        encoded_bytes = checked_add(
            encoded_bytes,
            checked_add(
                CHECKPOINT_CHUNK_HEADER_BYTES,
                payload_bytes,
                "checkpoint chunk record",
            )?,
            "checkpoint bytes",
        )?;
    }
    encoded_bytes = checked_add(
        encoded_bytes,
        CHECKPOINT_DIGEST_BYTES,
        "checkpoint terminal digest",
    )?;
    let capacity = usize_from_u64(encoded_bytes, "checkpoint allocation")?;
    let mut encoded = Vec::new();
    encoded
        .try_reserve_exact(capacity)
        .map_err(|_| ActivationCacheError::AllocationFailed {
            bytes: encoded_bytes,
        })?;

    encoded.extend_from_slice(&CHECKPOINT_MAGIC);
    encoded.extend_from_slice(&CHECKPOINT_VERSION.to_le_bytes());
    encoded.push(builder.spec.dtype.format_code());
    encoded.push(0);
    encoded.extend_from_slice(&builder.spec.layer_index.to_le_bytes());
    encoded.extend_from_slice(&builder.spec.total_tokens.to_le_bytes());
    encoded.extend_from_slice(&builder.spec.feature_width.to_le_bytes());
    encoded.extend_from_slice(&builder.spec.shard_tokens.to_le_bytes());
    encoded.extend_from_slice(builder.spec.source_digest.as_bytes());
    let name_len = u32::try_from(builder.spec.tensor_name.len()).map_err(|_| {
        ActivationCacheError::ArithmeticOverflow {
            context: "checkpoint tensor-name length",
        }
    })?;
    encoded.extend_from_slice(&name_len.to_le_bytes());
    encoded.extend_from_slice(builder.spec.tensor_name.as_bytes());
    let chunk_count = u32::try_from(builder.chunks.len()).map_err(|_| {
        ActivationCacheError::ArithmeticOverflow {
            context: "checkpoint chunk count",
        }
    })?;
    encoded.extend_from_slice(&chunk_count.to_le_bytes());
    encoded.extend_from_slice(&builder.covered_tokens.to_le_bytes());
    encoded.extend_from_slice(&0_u32.to_le_bytes());
    debug_assert_eq!(
        encoded.len() as u64,
        CHECKPOINT_MANIFEST_FIXED_BYTES + name_bytes
    );

    for chunk in &builder.chunks {
        let packed_mask = try_pack_mask(&chunk.token_mask)?;
        let value_len = len_u64(chunk.encoded_values.len(), "checkpoint value bytes")?;
        let mask_len = len_u64(packed_mask.len(), "checkpoint mask bytes")?;
        let boundary_count = len_u64(chunk.sequence_ends.len(), "checkpoint boundary count")?;
        encoded.extend_from_slice(&chunk.token_start.to_le_bytes());
        encoded.extend_from_slice(&chunk.token_count.to_le_bytes());
        encoded.extend_from_slice(&value_len.to_le_bytes());
        encoded.extend_from_slice(&mask_len.to_le_bytes());
        encoded.extend_from_slice(&boundary_count.to_le_bytes());
        encoded.extend_from_slice(chunk.digest.as_bytes());
        encoded.extend_from_slice(&chunk.encoded_values);
        encoded.extend_from_slice(&packed_mask);
        for boundary in &chunk.sequence_ends {
            encoded.extend_from_slice(&boundary.to_le_bytes());
        }
    }
    let digest = hash_domain(CHECKPOINT_DOMAIN, &encoded);
    encoded.extend_from_slice(digest.as_bytes());
    let got = len_u64(encoded.len(), "emitted checkpoint bytes")?;
    if got != encoded_bytes {
        return Err(ActivationCacheError::AccountingMismatch {
            expected: encoded_bytes,
            got,
        });
    }
    Ok((encoded, digest))
}

fn decode_builder_checkpoint(
    encoded: &[u8],
    expected_digest: Option<ActivationDigest>,
) -> Result<ActivationCacheBuilder, ActivationCacheError> {
    let minimum = CHECKPOINT_MANIFEST_FIXED_BYTES
        .checked_add(CHECKPOINT_DIGEST_BYTES)
        .ok_or(ActivationCacheError::ArithmeticOverflow {
            context: "minimum checkpoint bytes",
        })?;
    if len_u64(encoded.len(), "checkpoint input bytes")? < minimum {
        return Err(ActivationCacheError::TruncatedEncoding {
            context: "checkpoint manifest and digest",
            needed: minimum,
            remaining: len_u64(encoded.len(), "checkpoint input bytes")?,
        });
    }
    let digest_start = encoded.len() - CHECKPOINT_DIGEST_BYTES as usize;
    let (body, terminal) = encoded.split_at(digest_start);
    let persisted_digest = ActivationDigest::from_bytes(
        terminal
            .try_into()
            .expect("terminal checkpoint digest has fixed length"),
    );
    let got_digest = hash_domain(CHECKPOINT_DOMAIN, body);
    if persisted_digest != got_digest {
        return Err(ActivationCacheError::DigestMismatch {
            expected: persisted_digest,
            got: got_digest,
        });
    }
    if let Some(expected) = expected_digest
        && expected != got_digest
    {
        return Err(ActivationCacheError::DigestMismatch {
            expected,
            got: got_digest,
        });
    }

    let mut cursor = DecodeCursor::new(body);
    let magic = cursor.array::<4>("checkpoint magic")?;
    if magic != CHECKPOINT_MAGIC {
        return Err(ActivationCacheError::InvalidCheckpointMagic { got: magic });
    }
    let version = cursor.u16("checkpoint version")?;
    if version != CHECKPOINT_VERSION {
        return Err(ActivationCacheError::UnsupportedCheckpointVersion { got: version });
    }
    let dtype = decode_dtype(cursor.u8("checkpoint dtype")?)?;
    let reserved = cursor.u8("checkpoint reserved byte")?;
    if reserved != 0 {
        return Err(ActivationCacheError::NonZeroReservedByte { got: reserved });
    }
    let layer_index = cursor.u32("checkpoint layer index")?;
    let total_tokens = cursor.u64("checkpoint total tokens")?;
    let feature_width = cursor.u64("checkpoint feature width")?;
    let shard_tokens = cursor.u32("checkpoint shard tokens")?;
    let source_digest = ActivationDigest::from_bytes(cursor.array("checkpoint source digest")?);
    let name_len = u64::from(cursor.u32("checkpoint tensor-name length")?);
    let name_bytes = cursor.take(name_len, "checkpoint tensor name")?;
    let name =
        std::str::from_utf8(name_bytes).map_err(|_| ActivationCacheError::InvalidTensorNameUtf8)?;
    let mut tensor_name = String::new();
    tensor_name
        .try_reserve_exact(name_bytes.len())
        .map_err(|_| ActivationCacheError::AllocationFailed { bytes: name_len })?;
    tensor_name.push_str(name);
    let chunk_count = cursor.u32("checkpoint chunk count")?;
    let declared_covered_tokens = cursor.u64("checkpoint covered token count")?;
    let manifest_reserved = cursor.u32("checkpoint manifest reserved field")?;
    if manifest_reserved != 0 {
        let got = manifest_reserved
            .to_le_bytes()
            .into_iter()
            .find(|byte| *byte != 0)
            .expect("nonzero u32 has a nonzero byte");
        return Err(ActivationCacheError::NonZeroReservedByte { got });
    }
    debug_assert_eq!(
        cursor.position_u64()?,
        CHECKPOINT_MANIFEST_FIXED_BYTES + name_len
    );

    let spec = ActivationCacheSpec::new(
        layer_index,
        tensor_name,
        total_tokens,
        feature_width,
        dtype,
        source_digest,
        shard_tokens,
    )?;
    let minimum_headers = checked_mul(
        u64::from(chunk_count),
        CHECKPOINT_CHUNK_HEADER_BYTES,
        "checkpoint chunk headers",
    )?;
    if minimum_headers > cursor.remaining_u64()? {
        return Err(ActivationCacheError::TruncatedEncoding {
            context: "checkpoint chunk headers",
            needed: minimum_headers,
            remaining: cursor.remaining_u64()?,
        });
    }
    if u64::from(chunk_count) > spec.total_tokens {
        return Err(ActivationCacheError::CoveredTokenMismatch {
            expected: spec.total_tokens,
            got: u64::from(chunk_count),
        });
    }

    let mut builder = ActivationCacheBuilder::new(spec.clone());
    builder
        .chunks
        .try_reserve_exact(chunk_count as usize)
        .map_err(|_| ActivationCacheError::AllocationFailed {
            bytes: u64::from(chunk_count),
        })?;
    let bytes_per_token = checked_mul(
        spec.feature_width,
        u64::from(spec.dtype.encoded_width()),
        "checkpoint bytes per token",
    )?;
    let mut previous_start = None;
    for _ in 0..chunk_count {
        let token_start = cursor.u64("checkpoint chunk start")?;
        let token_count = cursor.u64("checkpoint chunk count")?;
        let value_len = cursor.u64("checkpoint chunk value bytes")?;
        let mask_len = cursor.u64("checkpoint chunk mask bytes")?;
        let boundary_count = cursor.u64("checkpoint chunk boundary count")?;
        let persisted_chunk_digest =
            ActivationDigest::from_bytes(cursor.array("checkpoint chunk digest")?);
        if let Some(previous) = previous_start
            && token_start <= previous
        {
            return Err(ActivationCacheError::NonCanonicalChunkOrder {
                previous_start: previous,
                start: token_start,
            });
        }
        previous_start = Some(token_start);
        if token_count == 0 {
            return Err(ActivationCacheError::EmptyChunk);
        }
        let token_end = token_start.checked_add(token_count).ok_or(
            ActivationCacheError::ArithmeticOverflow {
                context: "checkpoint chunk end",
            },
        )?;
        if token_start >= spec.total_tokens || token_end > spec.total_tokens {
            return Err(ActivationCacheError::ChunkOutOfBounds {
                start: token_start,
                count: token_count,
                total: spec.total_tokens,
            });
        }
        let expected_value_len =
            checked_mul(token_count, bytes_per_token, "checkpoint chunk value bytes")?;
        if value_len != expected_value_len {
            return Err(ActivationCacheError::ValueCountMismatch {
                expected: expected_value_len,
                got: value_len,
            });
        }
        let expected_mask_len = token_count.div_ceil(8);
        if mask_len != expected_mask_len {
            return Err(ActivationCacheError::MaskLengthMismatch {
                expected: expected_mask_len,
                got: mask_len,
            });
        }
        if boundary_count > token_count {
            return Err(ActivationCacheError::InvalidBoundary {
                previous: token_start,
                boundary: token_end.saturating_add(1),
                chunk_start: token_start,
                chunk_end: token_end,
            });
        }
        let boundary_bytes = checked_mul(boundary_count, 8, "checkpoint chunk boundary bytes")?;
        let payload_bytes = checked_add(
            checked_add(value_len, mask_len, "checkpoint values and mask")?,
            boundary_bytes,
            "checkpoint chunk payload",
        )?;
        if payload_bytes > cursor.remaining_u64()? {
            return Err(ActivationCacheError::TruncatedEncoding {
                context: "checkpoint chunk payload",
                needed: payload_bytes,
                remaining: cursor.remaining_u64()?,
            });
        }
        let encoded_values = cursor.take(value_len, "checkpoint chunk values")?;
        let scalar_start = checked_mul(token_start, spec.feature_width, "checkpoint scalar start")?;
        validate_encoded_values(spec.dtype, encoded_values, scalar_start)?;
        let packed_mask = cursor.take(mask_len, "checkpoint chunk mask")?;
        validate_canonical_mask_bits(packed_mask, token_count, 0)?;
        let token_capacity = usize_from_u64(token_count, "checkpoint mask allocation")?;
        let mut token_mask = Vec::new();
        token_mask
            .try_reserve_exact(token_capacity)
            .map_err(|_| ActivationCacheError::AllocationFailed { bytes: token_count })?;
        for index in 0..token_capacity {
            token_mask.push((packed_mask[index / 8] >> (index % 8)) & 1 != 0);
        }
        let boundary_capacity = usize_from_u64(boundary_count, "checkpoint boundary allocation")?;
        let mut sequence_ends = Vec::new();
        sequence_ends
            .try_reserve_exact(boundary_capacity)
            .map_err(|_| ActivationCacheError::AllocationFailed {
                bytes: boundary_bytes,
            })?;
        let mut previous_boundary = token_start;
        for _ in 0..boundary_count {
            let boundary = cursor.u64("checkpoint sequence boundary")?;
            if boundary <= previous_boundary || boundary > token_end {
                return Err(ActivationCacheError::InvalidBoundary {
                    previous: previous_boundary,
                    boundary,
                    chunk_start: token_start,
                    chunk_end: token_end,
                });
            }
            previous_boundary = boundary;
            sequence_ends.push(boundary);
        }
        let got_chunk_digest = compute_chunk_digest_encoded_mask(
            spec.schema_digest,
            token_start,
            token_count,
            encoded_values,
            packed_mask,
            &sequence_ends,
        );
        if got_chunk_digest != persisted_chunk_digest {
            return Err(ActivationCacheError::CheckpointChunkDigestMismatch {
                token_start,
                expected: persisted_chunk_digest,
                got: got_chunk_digest,
            });
        }
        let mut owned_values = Vec::new();
        owned_values
            .try_reserve_exact(encoded_values.len())
            .map_err(|_| ActivationCacheError::AllocationFailed { bytes: value_len })?;
        owned_values.extend_from_slice(encoded_values);
        builder.ingest(ActivationChunk {
            schema_digest: spec.schema_digest,
            token_start,
            token_count,
            encoded_values: owned_values,
            token_mask,
            sequence_ends,
            digest: got_chunk_digest,
        })?;
    }
    if cursor.remaining_u64()? != 0 {
        return Err(ActivationCacheError::TrailingBytes {
            count: cursor.remaining_u64()?,
        });
    }
    if builder.covered_tokens != declared_covered_tokens {
        return Err(ActivationCacheError::CoveredTokenMismatch {
            expected: builder.covered_tokens,
            got: declared_covered_tokens,
        });
    }
    Ok(builder)
}

/// Exact canonical byte accounting for a finalized activation cache.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ActivationByteLedger {
    manifest_bytes: u64,
    shard_header_bytes: u64,
    value_bytes: u64,
    mask_bytes: u64,
    boundary_bytes: u64,
    encoded_bytes: u64,
}

impl ActivationByteLedger {
    /// Returns bytes occupied by the top-level schema and shard manifest.
    pub const fn manifest_bytes(self) -> u64 {
        self.manifest_bytes
    }

    /// Returns bytes occupied by all fixed canonical shard headers.
    pub const fn shard_header_bytes(self) -> u64 {
        self.shard_header_bytes
    }

    /// Returns bytes occupied by dtype-encoded activation scalars.
    pub const fn value_bytes(self) -> u64 {
        self.value_bytes
    }

    /// Returns bytes occupied by canonical per-shard bit-packed token masks.
    pub const fn mask_bytes(self) -> u64 {
        self.mask_bytes
    }

    /// Returns bytes occupied by global exclusive sequence ends in shard payloads.
    pub const fn boundary_bytes(self) -> u64 {
        self.boundary_bytes
    }

    /// Returns the exact total length of the canonical artifact encoding.
    pub const fn encoded_bytes(self) -> u64 {
        self.encoded_bytes
    }

    /// Returns exact bytes in the cache's canonical backing store.
    ///
    /// This protocol ledger excludes Rust index structures used to navigate the backing store.
    pub const fn cache_bytes(self) -> u64 {
        self.encoded_bytes
    }
}

/// Index entry for one fixed-range canonical cache shard.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActivationShard {
    index: u32,
    token_start: u64,
    token_count: u32,
    sequence_ends: Vec<u64>,
    digest: ActivationDigest,
    encoded_offset: u64,
    encoded_bytes: u64,
    value_bytes: u64,
    mask_bytes: u64,
    boundary_bytes: u64,
}

impl ActivationShard {
    /// Returns the contiguous zero-based canonical shard index.
    pub const fn index(&self) -> u32 {
        self.index
    }

    /// Returns the inclusive global token offset of this shard.
    pub const fn token_start(&self) -> u64 {
        self.token_start
    }

    /// Returns the number of token rows in this shard.
    pub const fn token_count(&self) -> u32 {
        self.token_count
    }

    /// Returns the exclusive global token offset after this shard.
    pub const fn token_end(&self) -> u64 {
        self.token_start + self.token_count as u64
    }

    /// Returns exclusive sequence ends whose final token lies in this shard.
    pub fn sequence_ends(&self) -> &[u64] {
        &self.sequence_ends
    }

    /// Returns the domain-separated digest of this shard's complete canonical record.
    pub const fn digest(&self) -> ActivationDigest {
        self.digest
    }

    /// Returns the byte offset of this shard record in the complete artifact.
    pub const fn encoded_offset(&self) -> u64 {
        self.encoded_offset
    }

    /// Returns the exact complete encoded shard-record length.
    pub const fn encoded_bytes(&self) -> u64 {
        self.encoded_bytes
    }

    /// Returns dtype-encoded activation payload bytes in this shard.
    pub const fn value_bytes(&self) -> u64 {
        self.value_bytes
    }

    /// Returns bit-packed token-mask bytes in this shard.
    pub const fn mask_bytes(&self) -> u64 {
        self.mask_bytes
    }

    /// Returns sequence-boundary payload bytes in this shard.
    pub const fn boundary_bytes(&self) -> u64 {
        self.boundary_bytes
    }
}

/// Final canonical activation artifact with content-addressed shard indices.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActivationCache {
    spec: ActivationCacheSpec,
    digest: ActivationDigest,
    encoded: Vec<u8>,
    shards: Vec<ActivationShard>,
    sequence_ends: Vec<u64>,
    byte_ledger: ActivationByteLedger,
}

impl ActivationCache {
    /// Reopens and fully validates one canonical cache encoding.
    ///
    /// All manifest and shard fields, payload lengths, masks, scalar values, boundaries, and
    /// canonical ranges are checked before the backing bytes are copied. Schema, shard, and cache
    /// digests are recomputed from the persisted bytes.
    pub fn from_encoded(encoded: &[u8]) -> Result<Self, ActivationCacheError> {
        decode_activation_cache(encoded, None)
    }

    /// Reopens canonical bytes and also requires their recomputed identity to match `expected`.
    pub fn from_encoded_verified(
        encoded: &[u8],
        expected: ActivationDigest,
    ) -> Result<Self, ActivationCacheError> {
        decode_activation_cache(encoded, Some(expected))
    }

    /// Reads at most `max_encoded_bytes` from durable storage, then reopens the canonical cache.
    ///
    /// The explicit bound is enforced before growing the input buffer. It should come from the
    /// campaign receipt or another trusted storage policy, not from the file being decoded.
    pub fn read_from<R: Read>(
        reader: R,
        max_encoded_bytes: u64,
    ) -> Result<Self, ActivationCacheError> {
        let encoded = read_bounded(reader, max_encoded_bytes)?;
        decode_activation_cache_owned(encoded, None)
    }

    /// Reads a bounded canonical cache and verifies its recomputed content identity.
    ///
    /// The reader buffer becomes the cache backing store after validation; reopening therefore
    /// does not duplicate the complete artifact in memory.
    pub fn read_from_verified<R: Read>(
        reader: R,
        max_encoded_bytes: u64,
        expected: ActivationDigest,
    ) -> Result<Self, ActivationCacheError> {
        let encoded = read_bounded(reader, max_encoded_bytes)?;
        decode_activation_cache_owned(encoded, Some(expected))
    }

    /// Returns the schema and source provenance encoded by the artifact.
    pub const fn spec(&self) -> &ActivationCacheSpec {
        &self.spec
    }

    /// Returns the digest of the complete canonical artifact.
    pub const fn digest(&self) -> ActivationDigest {
        self.digest
    }

    /// Returns the complete canonical artifact encoding.
    pub fn encoded(&self) -> &[u8] {
        &self.encoded
    }

    /// Returns canonical shards in contiguous index order.
    pub fn shards(&self) -> &[ActivationShard] {
        &self.shards
    }

    /// Returns one complete canonical shard record by its zero-based index.
    pub fn encoded_shard(&self, index: u32) -> Option<&[u8]> {
        let shard = self.shards.get(usize::try_from(index).ok()?)?;
        let start = usize::try_from(shard.encoded_offset).ok()?;
        let end = shard
            .encoded_offset
            .checked_add(shard.encoded_bytes)
            .and_then(|value| usize::try_from(value).ok())?;
        self.encoded.get(start..end)
    }

    /// Returns every global exclusive sequence end in ascending order.
    pub fn sequence_ends(&self) -> &[u64] {
        &self.sequence_ends
    }

    /// Returns exact canonical component and total byte counts.
    pub const fn byte_ledger(&self) -> ActivationByteLedger {
        self.byte_ledger
    }
}

fn read_bounded<R: Read>(
    mut reader: R,
    max_encoded_bytes: u64,
) -> Result<Vec<u8>, ActivationCacheError> {
    let mut encoded = Vec::new();
    let mut buffer = [0_u8; 8 * 1024];
    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|error| ActivationCacheError::ReadFailed { kind: error.kind() })?;
        if read == 0 {
            break;
        }
        let current = len_u64(encoded.len(), "reader cache length")?;
        let next = checked_add(
            current,
            len_u64(read, "reader chunk length")?,
            "reader cache length",
        )?;
        if next > max_encoded_bytes {
            return Err(ActivationCacheError::EncodedSizeLimitExceeded {
                limit: max_encoded_bytes,
            });
        }
        encoded
            .try_reserve(read)
            .map_err(|_| ActivationCacheError::AllocationFailed { bytes: next })?;
        encoded.extend_from_slice(&buffer[..read]);
    }
    Ok(encoded)
}

fn decode_dtype(code: u8) -> Result<ActivationDType, ActivationCacheError> {
    match code {
        1 => Ok(ActivationDType::Float32),
        2 => Ok(ActivationDType::Float16),
        3 => Ok(ActivationDType::BFloat16),
        got => Err(ActivationCacheError::InvalidDTypeCode { got }),
    }
}

struct DecodeCursor<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> DecodeCursor<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn take(
        &mut self,
        count: u64,
        context: &'static str,
    ) -> Result<&'a [u8], ActivationCacheError> {
        let remaining = len_u64(self.bytes.len() - self.position, "decoder remaining bytes")?;
        if count > remaining {
            return Err(ActivationCacheError::TruncatedEncoding {
                context,
                needed: count,
                remaining,
            });
        }
        let count = usize_from_u64(count, "decoder field length")?;
        let end =
            self.position
                .checked_add(count)
                .ok_or(ActivationCacheError::ArithmeticOverflow {
                    context: "decoder field end",
                })?;
        let value = &self.bytes[self.position..end];
        self.position = end;
        Ok(value)
    }

    fn array<const N: usize>(
        &mut self,
        context: &'static str,
    ) -> Result<[u8; N], ActivationCacheError> {
        let bytes = self.take(N as u64, context)?;
        let mut value = [0_u8; N];
        value.copy_from_slice(bytes);
        Ok(value)
    }

    fn u8(&mut self, context: &'static str) -> Result<u8, ActivationCacheError> {
        Ok(self.array::<1>(context)?[0])
    }

    fn u16(&mut self, context: &'static str) -> Result<u16, ActivationCacheError> {
        Ok(u16::from_le_bytes(self.array(context)?))
    }

    fn u32(&mut self, context: &'static str) -> Result<u32, ActivationCacheError> {
        Ok(u32::from_le_bytes(self.array(context)?))
    }

    fn u64(&mut self, context: &'static str) -> Result<u64, ActivationCacheError> {
        Ok(u64::from_le_bytes(self.array(context)?))
    }

    fn position_u64(&self) -> Result<u64, ActivationCacheError> {
        len_u64(self.position, "decoder position")
    }

    fn remaining_u64(&self) -> Result<u64, ActivationCacheError> {
        len_u64(self.bytes.len() - self.position, "decoder trailing bytes")
    }
}

fn decode_activation_cache(
    encoded: &[u8],
    expected_digest: Option<ActivationDigest>,
) -> Result<ActivationCache, ActivationCacheError> {
    let decoded = parse_activation_cache(encoded, expected_digest)?;
    let encoded_bytes = len_u64(encoded.len(), "decoded cache bytes")?;
    let mut owned = Vec::new();
    owned
        .try_reserve_exact(encoded.len())
        .map_err(|_| ActivationCacheError::AllocationFailed {
            bytes: encoded_bytes,
        })?;
    owned.extend_from_slice(encoded);
    Ok(decoded.with_encoding(owned))
}

fn decode_activation_cache_owned(
    encoded: Vec<u8>,
    expected_digest: Option<ActivationDigest>,
) -> Result<ActivationCache, ActivationCacheError> {
    let decoded = parse_activation_cache(&encoded, expected_digest)?;
    Ok(decoded.with_encoding(encoded))
}

struct DecodedActivationCache {
    spec: ActivationCacheSpec,
    digest: ActivationDigest,
    shards: Vec<ActivationShard>,
    sequence_ends: Vec<u64>,
    byte_ledger: ActivationByteLedger,
}

impl DecodedActivationCache {
    fn with_encoding(self, encoded: Vec<u8>) -> ActivationCache {
        ActivationCache {
            spec: self.spec,
            digest: self.digest,
            encoded,
            shards: self.shards,
            sequence_ends: self.sequence_ends,
            byte_ledger: self.byte_ledger,
        }
    }
}

fn parse_activation_cache(
    encoded: &[u8],
    expected_digest: Option<ActivationDigest>,
) -> Result<DecodedActivationCache, ActivationCacheError> {
    let mut cursor = DecodeCursor::new(encoded);
    let magic = cursor.array::<4>("cache magic")?;
    if magic != CACHE_MAGIC {
        return Err(ActivationCacheError::InvalidCacheMagic { got: magic });
    }
    let version = cursor.u16("cache version")?;
    if version != CACHE_VERSION {
        return Err(ActivationCacheError::UnsupportedCacheVersion { got: version });
    }
    let dtype_code = cursor.u8("cache dtype")?;
    let dtype = match dtype_code {
        1 => ActivationDType::Float32,
        2 => ActivationDType::Float16,
        3 => ActivationDType::BFloat16,
        got => return Err(ActivationCacheError::InvalidDTypeCode { got }),
    };
    let reserved = cursor.u8("manifest reserved byte")?;
    if reserved != 0 {
        return Err(ActivationCacheError::NonZeroReservedByte { got: reserved });
    }
    let layer_index = cursor.u32("layer index")?;
    let total_tokens = cursor.u64("total token count")?;
    let feature_width = cursor.u64("feature width")?;
    let shard_tokens = cursor.u32("shard token count")?;
    let source_digest = ActivationDigest::from_bytes(cursor.array("source digest")?);
    let tensor_name_len = u64::from(cursor.u32("tensor-name length")?);
    let tensor_name_bytes = cursor.take(tensor_name_len, "tensor name")?;
    let tensor_name_str = std::str::from_utf8(tensor_name_bytes)
        .map_err(|_| ActivationCacheError::InvalidTensorNameUtf8)?;
    let mut tensor_name = String::new();
    tensor_name
        .try_reserve_exact(tensor_name_bytes.len())
        .map_err(|_| ActivationCacheError::AllocationFailed {
            bytes: tensor_name_len,
        })?;
    tensor_name.push_str(tensor_name_str);
    let persisted_shard_count = cursor.u32("shard count")?;
    let manifest_bytes = cursor.position_u64()?;

    let spec = ActivationCacheSpec::new(
        layer_index,
        tensor_name,
        total_tokens,
        feature_width,
        dtype,
        source_digest,
        shard_tokens,
    )?;
    if persisted_shard_count != spec.shard_count {
        return Err(ActivationCacheError::ShardCountMismatch {
            expected: spec.shard_count,
            got: persisted_shard_count,
        });
    }
    let minimum_header_bytes = checked_mul(
        u64::from(persisted_shard_count),
        SHARD_HEADER_BYTES,
        "minimum persisted shard headers",
    )?;
    if minimum_header_bytes > cursor.remaining_u64()? {
        return Err(ActivationCacheError::TruncatedEncoding {
            context: "shard headers",
            needed: minimum_header_bytes,
            remaining: cursor.remaining_u64()?,
        });
    }

    let shard_capacity =
        usize_from_u64(u64::from(persisted_shard_count), "decoded shard allocation")?;
    let mut shards = Vec::new();
    shards.try_reserve_exact(shard_capacity).map_err(|_| {
        ActivationCacheError::AllocationFailed {
            bytes: u64::from(persisted_shard_count),
        }
    })?;
    let mut sequence_ends = Vec::new();
    let bytes_per_token = checked_mul(
        spec.feature_width,
        u64::from(spec.dtype.encoded_width()),
        "decoded bytes per token",
    )?;
    let mut value_bytes = 0_u64;
    let mut mask_bytes = 0_u64;
    let mut boundary_bytes = 0_u64;

    for expected_index in 0..persisted_shard_count {
        let record_start = cursor.position;
        let got_index = cursor.u32("shard index")?;
        if got_index != expected_index {
            return Err(ActivationCacheError::ShardIndexMismatch {
                expected: expected_index,
                got: got_index,
            });
        }
        let got_start = cursor.u64("shard token start")?;
        let got_count = cursor.u32("shard token count")?;
        let got_value_bytes = cursor.u64("shard value bytes")?;
        let got_mask_bytes = u64::from(cursor.u32("shard mask bytes")?);
        let boundary_count = cursor.u32("shard boundary count")?;

        let expected_start = checked_mul(
            u64::from(expected_index),
            u64::from(spec.shard_tokens),
            "decoded shard token start",
        )?;
        let expected_count_u64 =
            (spec.total_tokens - expected_start).min(u64::from(spec.shard_tokens));
        let expected_count = u32::try_from(expected_count_u64).map_err(|_| {
            ActivationCacheError::ArithmeticOverflow {
                context: "decoded shard token count",
            }
        })?;
        if got_start != expected_start || got_count != expected_count {
            return Err(ActivationCacheError::ShardRangeMismatch {
                index: expected_index,
                expected_start,
                got_start,
                expected_count,
                got_count,
            });
        }
        let expected_value_bytes = checked_mul(
            expected_count_u64,
            bytes_per_token,
            "decoded shard value bytes",
        )?;
        if got_value_bytes != expected_value_bytes {
            return Err(ActivationCacheError::ShardLengthMismatch {
                index: expected_index,
                component: "value",
                expected: expected_value_bytes,
                got: got_value_bytes,
            });
        }
        let expected_mask_bytes = expected_count_u64.div_ceil(8);
        if got_mask_bytes != expected_mask_bytes {
            return Err(ActivationCacheError::ShardLengthMismatch {
                index: expected_index,
                component: "mask",
                expected: expected_mask_bytes,
                got: got_mask_bytes,
            });
        }
        if u64::from(boundary_count) > expected_count_u64 {
            return Err(ActivationCacheError::ShardLengthMismatch {
                index: expected_index,
                component: "boundary count",
                expected: expected_count_u64,
                got: u64::from(boundary_count),
            });
        }
        let shard_boundary_bytes =
            checked_mul(u64::from(boundary_count), 8, "decoded shard boundary bytes")?;
        let payload_bytes = checked_add(
            checked_add(
                expected_value_bytes,
                expected_mask_bytes,
                "decoded shard values and mask",
            )?,
            shard_boundary_bytes,
            "decoded shard payload",
        )?;
        if payload_bytes > cursor.remaining_u64()? {
            return Err(ActivationCacheError::TruncatedEncoding {
                context: "shard payload",
                needed: payload_bytes,
                remaining: cursor.remaining_u64()?,
            });
        }

        let shard_values = cursor.take(expected_value_bytes, "shard values")?;
        let scalar_start = checked_mul(expected_start, spec.feature_width, "decoded scalar start")?;
        validate_encoded_values(spec.dtype, shard_values, scalar_start)?;
        let shard_mask = cursor.take(expected_mask_bytes, "shard mask")?;
        validate_canonical_mask(shard_mask, expected_count, expected_index)?;

        let boundary_capacity =
            usize_from_u64(u64::from(boundary_count), "decoded boundary allocation")?;
        let mut shard_sequence_ends = Vec::new();
        shard_sequence_ends
            .try_reserve_exact(boundary_capacity)
            .map_err(|_| ActivationCacheError::AllocationFailed {
                bytes: shard_boundary_bytes,
            })?;
        sequence_ends
            .try_reserve_exact(boundary_capacity)
            .map_err(|_| ActivationCacheError::AllocationFailed {
                bytes: shard_boundary_bytes,
            })?;
        let token_end = checked_add(
            expected_start,
            expected_count_u64,
            "decoded shard token end",
        )?;
        let mut previous_boundary = expected_start;
        for _ in 0..boundary_count {
            let boundary = cursor.u64("sequence boundary")?;
            if boundary <= previous_boundary || boundary > token_end {
                return Err(ActivationCacheError::InvalidBoundary {
                    previous: previous_boundary,
                    boundary,
                    chunk_start: expected_start,
                    chunk_end: token_end,
                });
            }
            previous_boundary = boundary;
            shard_sequence_ends.push(boundary);
            sequence_ends.push(boundary);
        }

        let record_end = cursor.position;
        let encoded_offset = len_u64(record_start, "decoded shard offset")?;
        let encoded_bytes = len_u64(record_end - record_start, "decoded shard bytes")?;
        let digest = hash_shard(spec.schema_digest, &encoded[record_start..record_end]);
        shards.push(ActivationShard {
            index: expected_index,
            token_start: expected_start,
            token_count: expected_count,
            sequence_ends: shard_sequence_ends,
            digest,
            encoded_offset,
            encoded_bytes,
            value_bytes: expected_value_bytes,
            mask_bytes: expected_mask_bytes,
            boundary_bytes: shard_boundary_bytes,
        });
        value_bytes = checked_add(value_bytes, expected_value_bytes, "decoded value ledger")?;
        mask_bytes = checked_add(mask_bytes, expected_mask_bytes, "decoded mask ledger")?;
        boundary_bytes = checked_add(
            boundary_bytes,
            shard_boundary_bytes,
            "decoded boundary ledger",
        )?;
    }

    let trailing = cursor.remaining_u64()?;
    if trailing != 0 {
        return Err(ActivationCacheError::TrailingBytes { count: trailing });
    }
    if sequence_ends.last().copied() != Some(spec.total_tokens) {
        return Err(ActivationCacheError::MissingTerminalBoundary {
            total_tokens: spec.total_tokens,
        });
    }

    let shard_header_bytes = checked_mul(
        u64::from(spec.shard_count),
        SHARD_HEADER_BYTES,
        "decoded shard header ledger",
    )?;
    let encoded_bytes = len_u64(encoded.len(), "decoded cache bytes")?;
    let accounted_bytes = [
        manifest_bytes,
        shard_header_bytes,
        value_bytes,
        mask_bytes,
        boundary_bytes,
    ]
    .into_iter()
    .try_fold(0_u64, |total, bytes| {
        checked_add(total, bytes, "decoded cache ledger")
    })?;
    if accounted_bytes != encoded_bytes {
        return Err(ActivationCacheError::AccountingMismatch {
            expected: accounted_bytes,
            got: encoded_bytes,
        });
    }
    let digest = hash_domain(CACHE_DOMAIN, encoded);
    if let Some(expected) = expected_digest
        && digest != expected
    {
        return Err(ActivationCacheError::DigestMismatch {
            expected,
            got: digest,
        });
    }
    Ok(DecodedActivationCache {
        spec,
        digest,
        shards,
        sequence_ends,
        byte_ledger: ActivationByteLedger {
            manifest_bytes,
            shard_header_bytes,
            value_bytes,
            mask_bytes,
            boundary_bytes,
            encoded_bytes,
        },
    })
}

fn validate_encoded_values(
    dtype: ActivationDType,
    encoded: &[u8],
    scalar_start: u64,
) -> Result<(), ActivationCacheError> {
    let width = usize::from(dtype.encoded_width());
    for (offset, bytes) in encoded.chunks_exact(width).enumerate() {
        let finite = match dtype {
            ActivationDType::Float32 => {
                let mut value = [0_u8; 4];
                value.copy_from_slice(bytes);
                f32::from_bits(u32::from_le_bytes(value)).is_finite()
            }
            ActivationDType::Float16 => {
                let mut value = [0_u8; 2];
                value.copy_from_slice(bytes);
                f16::from_bits(u16::from_le_bytes(value)).is_finite()
            }
            ActivationDType::BFloat16 => {
                let mut value = [0_u8; 2];
                value.copy_from_slice(bytes);
                bf16::from_bits(u16::from_le_bytes(value)).is_finite()
            }
        };
        if !finite {
            return Err(ActivationCacheError::NonFiniteEncodedValue {
                index: checked_add(
                    scalar_start,
                    len_u64(offset, "decoded scalar offset")?,
                    "decoded scalar index",
                )?,
            });
        }
    }
    Ok(())
}

fn validate_canonical_mask(
    encoded: &[u8],
    token_count: u32,
    shard_index: u32,
) -> Result<(), ActivationCacheError> {
    validate_canonical_mask_bits(encoded, u64::from(token_count), shard_index)
}

fn validate_canonical_mask_bits(
    encoded: &[u8],
    token_count: u64,
    shard_index: u32,
) -> Result<(), ActivationCacheError> {
    let used_terminal_bits = (token_count % 8) as u32;
    if used_terminal_bits != 0 {
        let valid_bits = ((1_u16 << used_terminal_bits) - 1) as u8;
        if encoded.last().is_some_and(|byte| byte & !valid_bits != 0) {
            return Err(ActivationCacheError::NonCanonicalMask { index: shard_index });
        }
    }
    Ok(())
}

fn encode_cache(
    spec: ActivationCacheSpec,
    values: Vec<u8>,
    token_mask: Vec<bool>,
    sequence_ends: Vec<u64>,
) -> Result<ActivationCache, ActivationCacheError> {
    let manifest_bytes = checked_add(
        CACHE_MANIFEST_FIXED_BYTES,
        len_u64(spec.tensor_name.len(), "manifest tensor-name bytes")?,
        "manifest bytes",
    )?;
    let shard_header_bytes = checked_mul(
        u64::from(spec.shard_count),
        SHARD_HEADER_BYTES,
        "shard header bytes",
    )?;
    let value_bytes = len_u64(values.len(), "cache value bytes")?;
    let boundary_bytes = checked_mul(
        len_u64(sequence_ends.len(), "sequence boundary count")?,
        8,
        "sequence boundary bytes",
    )?;
    let mut mask_bytes = 0_u64;
    for index in 0..spec.shard_count {
        let token_start = checked_mul(
            u64::from(index),
            u64::from(spec.shard_tokens),
            "shard token start",
        )?;
        let token_count = (spec.total_tokens - token_start).min(u64::from(spec.shard_tokens));
        mask_bytes = checked_add(mask_bytes, token_count.div_ceil(8), "mask bytes")?;
    }
    let encoded_bytes = [
        manifest_bytes,
        shard_header_bytes,
        value_bytes,
        mask_bytes,
        boundary_bytes,
    ]
    .into_iter()
    .try_fold(0_u64, |total, bytes| {
        checked_add(total, bytes, "total encoded cache bytes")
    })?;
    let capacity = usize_from_u64(encoded_bytes, "canonical cache allocation")?;
    let mut encoded = Vec::new();
    encoded
        .try_reserve_exact(capacity)
        .map_err(|_| ActivationCacheError::AllocationFailed {
            bytes: encoded_bytes,
        })?;

    encoded.extend_from_slice(&CACHE_MAGIC);
    encoded.extend_from_slice(&CACHE_VERSION.to_le_bytes());
    encoded.push(spec.dtype.format_code());
    encoded.push(0);
    encoded.extend_from_slice(&spec.layer_index.to_le_bytes());
    encoded.extend_from_slice(&spec.total_tokens.to_le_bytes());
    encoded.extend_from_slice(&spec.feature_width.to_le_bytes());
    encoded.extend_from_slice(&spec.shard_tokens.to_le_bytes());
    encoded.extend_from_slice(spec.source_digest.as_bytes());
    let tensor_name_len = u32::try_from(spec.tensor_name.len()).map_err(|_| {
        ActivationCacheError::ArithmeticOverflow {
            context: "manifest tensor-name length",
        }
    })?;
    encoded.extend_from_slice(&tensor_name_len.to_le_bytes());
    encoded.extend_from_slice(spec.tensor_name.as_bytes());
    encoded.extend_from_slice(&spec.shard_count.to_le_bytes());

    let bytes_per_token = checked_mul(
        spec.feature_width,
        u64::from(spec.dtype.encoded_width()),
        "bytes per token",
    )?;
    let mut shards = Vec::new();
    let shard_capacity = usize::try_from(spec.shard_count).map_err(|_| {
        ActivationCacheError::ArithmeticOverflow {
            context: "shard index allocation",
        }
    })?;
    shards.try_reserve_exact(shard_capacity).map_err(|_| {
        ActivationCacheError::AllocationFailed {
            bytes: u64::from(spec.shard_count),
        }
    })?;
    let mut boundary_cursor = 0usize;
    for index in 0..spec.shard_count {
        let token_start = checked_mul(
            u64::from(index),
            u64::from(spec.shard_tokens),
            "shard token start",
        )?;
        let token_count_u64 = (spec.total_tokens - token_start).min(u64::from(spec.shard_tokens));
        let token_count = u32::try_from(token_count_u64).map_err(|_| {
            ActivationCacheError::ArithmeticOverflow {
                context: "shard token count",
            }
        })?;
        let token_end = checked_add(token_start, token_count_u64, "shard token end")?;
        let value_start = checked_mul(token_start, bytes_per_token, "shard value start")?;
        let shard_value_bytes = checked_mul(token_count_u64, bytes_per_token, "shard value bytes")?;
        let value_end = checked_add(value_start, shard_value_bytes, "shard value end")?;
        let value_start = usize_from_u64(value_start, "shard value start")?;
        let value_end = usize_from_u64(value_end, "shard value end")?;
        let mask_start = usize_from_u64(token_start, "shard mask start")?;
        let mask_end = usize_from_u64(token_end, "shard mask end")?;
        let packed_mask = try_pack_mask(&token_mask[mask_start..mask_end])?;
        let shard_mask_bytes = len_u64(packed_mask.len(), "shard mask bytes")?;
        let boundary_start = boundary_cursor;
        while boundary_cursor < sequence_ends.len() && sequence_ends[boundary_cursor] <= token_end {
            boundary_cursor += 1;
        }
        let shard_boundaries = &sequence_ends[boundary_start..boundary_cursor];
        let mut shard_sequence_ends = Vec::new();
        shard_sequence_ends
            .try_reserve_exact(shard_boundaries.len())
            .map_err(|_| ActivationCacheError::AllocationFailed {
                bytes: len_u64(shard_boundaries.len(), "shard boundary allocation")
                    .and_then(|count| checked_mul(count, 8, "shard boundary allocation"))
                    .unwrap_or(u64::MAX),
            })?;
        shard_sequence_ends.extend_from_slice(shard_boundaries);
        let shard_boundary_bytes = checked_mul(
            len_u64(shard_sequence_ends.len(), "shard boundary count")?,
            8,
            "shard boundary bytes",
        )?;

        let record_start = encoded.len();
        let encoded_offset = len_u64(record_start, "encoded shard offset")?;
        encoded.extend_from_slice(&index.to_le_bytes());
        encoded.extend_from_slice(&token_start.to_le_bytes());
        encoded.extend_from_slice(&token_count.to_le_bytes());
        encoded.extend_from_slice(&shard_value_bytes.to_le_bytes());
        let mask_len = u32::try_from(shard_mask_bytes).map_err(|_| {
            ActivationCacheError::ArithmeticOverflow {
                context: "shard mask length",
            }
        })?;
        encoded.extend_from_slice(&mask_len.to_le_bytes());
        let boundary_count = u32::try_from(shard_sequence_ends.len()).map_err(|_| {
            ActivationCacheError::ArithmeticOverflow {
                context: "shard boundary count",
            }
        })?;
        encoded.extend_from_slice(&boundary_count.to_le_bytes());
        encoded.extend_from_slice(&values[value_start..value_end]);
        encoded.extend_from_slice(&packed_mask);
        for boundary in &shard_sequence_ends {
            encoded.extend_from_slice(&boundary.to_le_bytes());
        }
        let record = &encoded[record_start..];
        let digest = hash_shard(spec.schema_digest, record);
        let record_bytes = len_u64(record.len(), "encoded shard record bytes")?;
        shards.push(ActivationShard {
            index,
            token_start,
            token_count,
            sequence_ends: shard_sequence_ends,
            digest,
            encoded_offset,
            encoded_bytes: record_bytes,
            value_bytes: shard_value_bytes,
            mask_bytes: shard_mask_bytes,
            boundary_bytes: shard_boundary_bytes,
        });
    }
    debug_assert_eq!(boundary_cursor, sequence_ends.len());

    let got = len_u64(encoded.len(), "emitted canonical cache bytes")?;
    if got != encoded_bytes {
        return Err(ActivationCacheError::AccountingMismatch {
            expected: encoded_bytes,
            got,
        });
    }
    let digest = hash_domain(CACHE_DOMAIN, &encoded);
    Ok(ActivationCache {
        spec,
        digest,
        encoded,
        shards,
        sequence_ends,
        byte_ledger: ActivationByteLedger {
            manifest_bytes,
            shard_header_bytes,
            value_bytes,
            mask_bytes,
            boundary_bytes,
            encoded_bytes,
        },
    })
}

fn encode_value(
    dtype: ActivationDType,
    value: f32,
    index: u64,
    output: &mut Vec<u8>,
) -> Result<(), ActivationCacheError> {
    if !value.is_finite() {
        return Err(ActivationCacheError::NonFiniteValue { index });
    }
    match dtype {
        ActivationDType::Float32 => output.extend_from_slice(&value.to_bits().to_le_bytes()),
        ActivationDType::Float16 => {
            let encoded = f16::from_f32(value);
            if !encoded.is_finite() {
                return Err(ActivationCacheError::DTypeOverflow { index, dtype });
            }
            output.extend_from_slice(&encoded.to_bits().to_le_bytes());
        }
        ActivationDType::BFloat16 => {
            let encoded = bf16::from_f32(value);
            if !encoded.is_finite() {
                return Err(ActivationCacheError::DTypeOverflow { index, dtype });
            }
            output.extend_from_slice(&encoded.to_bits().to_le_bytes());
        }
    }
    Ok(())
}

fn compute_schema_digest(
    layer_index: u32,
    tensor_name: &str,
    total_tokens: u64,
    feature_width: u64,
    dtype: ActivationDType,
    source_digest: ActivationDigest,
    shard_tokens: u32,
) -> ActivationDigest {
    let mut hasher = blake3::Hasher::new();
    hasher.update(SCHEMA_DOMAIN);
    hasher.update(&layer_index.to_le_bytes());
    hasher.update(&total_tokens.to_le_bytes());
    hasher.update(&feature_width.to_le_bytes());
    hasher.update(&[dtype.format_code()]);
    hasher.update(source_digest.as_bytes());
    hasher.update(&shard_tokens.to_le_bytes());
    hasher.update(&(tensor_name.len() as u64).to_le_bytes());
    hasher.update(tensor_name.as_bytes());
    ActivationDigest::from_bytes(*hasher.finalize().as_bytes())
}

fn compute_chunk_digest(
    schema_digest: ActivationDigest,
    token_start: u64,
    token_count: u64,
    encoded_values: &[u8],
    token_mask: &[bool],
    sequence_ends: &[u64],
) -> Result<ActivationDigest, ActivationCacheError> {
    let packed_mask = try_pack_mask(token_mask)?;
    Ok(compute_chunk_digest_encoded_mask(
        schema_digest,
        token_start,
        token_count,
        encoded_values,
        &packed_mask,
        sequence_ends,
    ))
}

fn compute_chunk_digest_encoded_mask(
    schema_digest: ActivationDigest,
    token_start: u64,
    token_count: u64,
    encoded_values: &[u8],
    packed_mask: &[u8],
    sequence_ends: &[u64],
) -> ActivationDigest {
    let mut hasher = blake3::Hasher::new();
    hasher.update(CHUNK_DOMAIN);
    hasher.update(schema_digest.as_bytes());
    hasher.update(&token_start.to_le_bytes());
    hasher.update(&token_count.to_le_bytes());
    hasher.update(&(encoded_values.len() as u64).to_le_bytes());
    hasher.update(encoded_values);
    hasher.update(&(packed_mask.len() as u64).to_le_bytes());
    hasher.update(packed_mask);
    hasher.update(&(sequence_ends.len() as u64).to_le_bytes());
    for boundary in sequence_ends {
        hasher.update(&boundary.to_le_bytes());
    }
    ActivationDigest::from_bytes(*hasher.finalize().as_bytes())
}

fn hash_domain(domain: &[u8], bytes: &[u8]) -> ActivationDigest {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(bytes);
    ActivationDigest::from_bytes(*hasher.finalize().as_bytes())
}

fn hash_shard(schema_digest: ActivationDigest, record: &[u8]) -> ActivationDigest {
    let mut hasher = blake3::Hasher::new();
    hasher.update(SHARD_DOMAIN);
    hasher.update(schema_digest.as_bytes());
    hasher.update(record);
    ActivationDigest::from_bytes(*hasher.finalize().as_bytes())
}

fn try_pack_mask(mask: &[bool]) -> Result<Vec<u8>, ActivationCacheError> {
    let packed_len = mask.len().div_ceil(8);
    let packed_bytes = len_u64(packed_len, "packed mask allocation")?;
    let mut packed = Vec::new();
    packed
        .try_reserve_exact(packed_len)
        .map_err(|_| ActivationCacheError::AllocationFailed {
            bytes: packed_bytes,
        })?;
    packed.resize(packed_len, 0);
    for (index, selected) in mask.iter().copied().enumerate() {
        if selected {
            packed[index / 8] |= 1 << (index % 8);
        }
    }
    Ok(packed)
}

fn checked_add(left: u64, right: u64, context: &'static str) -> Result<u64, ActivationCacheError> {
    left.checked_add(right)
        .ok_or(ActivationCacheError::ArithmeticOverflow { context })
}

fn checked_mul(left: u64, right: u64, context: &'static str) -> Result<u64, ActivationCacheError> {
    left.checked_mul(right)
        .ok_or(ActivationCacheError::ArithmeticOverflow { context })
}

fn len_u64(length: usize, context: &'static str) -> Result<u64, ActivationCacheError> {
    u64::try_from(length).map_err(|_| ActivationCacheError::ArithmeticOverflow { context })
}

fn usize_from_u64(value: u64, context: &'static str) -> Result<usize, ActivationCacheError> {
    usize::try_from(value).map_err(|_| ActivationCacheError::ArithmeticOverflow { context })
}
