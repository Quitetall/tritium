# Tritium brand assets

The mark is the representation: three bars for the three trit states, offset by their
value. `+1` runs full width, `0` is short, `-1` is pushed left. The palette is not
decorative — each colour *is* a state, so do not re-tint the bars per theme or per
surface.

| token | hex | use |
|---|---|---|
| `+1` | `#22D3EE` | top bar |
| `0` | `#E8ECEF` | middle bar (`#94A3B8` on the dark social ground) |
| `-1` | `#3B82F6` | bottom bar |
| ground | `#0B0F17` | social card, avatar |
| wordmark | `#F2F4F6` on dark · `#0B0F17` on light | |

Type: **JetBrains Mono 500**, tracking `-0.04em`.

## Files

| file | use |
|---|---|
| `tritium-icon.svg` | mark only, transparent — favicon and small contexts |
| `tritium-header-dark.svg` / `-light.svg` | mark + wordmark for the README header |
| `tritium-header-dark.png` | raster header, for surfaces that cannot take SVG |
| `tritium-avatar-512.svg` / `tritium-avatar-1024.png` | square org/repo avatar |
| `tritium-social-1280x640.svg` / `tritium-social-2560x1280.png` | GitHub social preview |

## Two things worth knowing before you edit these

**The header ships as a light/dark pair.** The wordmark is near-white, so a single
transparent header is invisible on GitHub's light theme. `README.md` selects between
them with `<picture>` + `prefers-color-scheme`; the *only* difference between the two
files is the `<text>` fill. If you regenerate one, regenerate both.

**The SVGs name `'JetBrains Mono'` and that is deliberate.** Rasterise with whatever
build is installed locally — substitute the family at render time rather than editing
the source, so the checked-in vectors stay correct for anyone who has the real font:

```sh
sed "s/'JetBrains Mono'/'JetBrainsMono Nerd Font'/g" tritium-social-1280x640.svg > /tmp/r.svg
rsvg-convert -w 2560 -h 1280 /tmp/r.svg -o tritium-social-2560x1280.png
```

## Applying the social preview

GitHub has no API for this. Upload `tritium-social-2560x1280.png` by hand at
**Settings → General → Social preview**.

## Provenance

Generated with Claude Design (project *Tritium logo design options*). The originals
carry C2PA content-credential manifests; those manifests are ~14 KB of base64 against
a few hundred bytes of geometry, so the copies here are stripped for web delivery. The
provenanced originals remain in the design project.
