# Product Positioning — the post-CamScanner slot

## The wave that passed
CamScanner-era (2011–2021): capture + cleanup. Pixels in → prettier pixels out.
That wave crested when cleanup became a commodity camera feature. "Scan" stopped
being magic.

## The wave now
Pixels in → DATA out. Point a camera at paper, get the numbers, the totals, the
structured fields. The scanner that READS.

## What we have that fits
- A 270M model (trained on one gaming GPU, zero API cost) that reads printed
  Indian paperwork: receipts, bank statements, card statements, electricity
  bills, telecom invoices, salary slips, tabular sheets.
- Honest refusal: "illegible — human needed" instead of a guess.
- Measured generalization: held-out institutions (LOIO), published in the eval
  table. Nobody else shows this at this size.

## Positioning statement (public demo)
"A 270M model that reads Indian retail paperwork. Runs on one gaming GPU.
No cloud, no per-document fee, no data leaving the machine. It says
'illegible' instead of guessing, and here are its numbers on institutions
it never saw."

NOT "general-purpose document AI" — that invites comparison with giants.

## Distribution ladder ($0)
1. Local demo: FastAPI on the 3060 + browser camera capture (built tonight,
   artifacts/paper_rig/demo/). Works on LAN + ngrok tunnel.
2. Model card on HuggingFace: weights + eval table + factory story (files only,
   zero compute).
3. Later: quantized ONNX/WASM in-browser build (~300MB int8) — the full
   "nothing leaves your device" story; also the Ashva deployment shape.

## The capture edge (why a web app beats a raw endpoint)
Burst capture + sharpness scoring in the wrapper: the model sees only the best
frame of a handheld shot. CamScanner's real moat was capture UX, not models —
we inherit that lesson directly.

## Wedges (vertical first, general never)
- Small business: receipt/bill → Tally-ready fields (offline, GST-aware).
- Individual: statement → month summary.
- Private (Ashva): clinical forms — never in the public demo.

## Morning proof gate
Before ANY public claim: render 5 fresh pages (unseen seeds), photograph them
like a phone would (motion blur, glare via the degradation layer), and check
answers + refusals. Publish only numbers that survive that.
