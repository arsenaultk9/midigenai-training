from midigenai import load_from_hub

gen = load_from_hub()                       # default checkpoint from the Hub
prompt = gen.encode_midi_file("riff.mid")
gen.generate_to_midi(prompt, "continuation.mid", max_new_tokens=512)