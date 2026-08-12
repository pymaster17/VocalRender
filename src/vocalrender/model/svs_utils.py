import torch
import torch.nn as nn
import math
from transformers import PreTrainedTokenizer

def get_svs_token_maps():
    """Returns mapping for SVS tokens."""
    pitch_tokens = [f"<P_{i}>" for i in range(128)]
    base_note_tokens = ["<NOTE_1>", "<NOTE_2>", "<NOTE_4>", "<NOTE_8>", "<NOTE_16>", "<NOTE_32>"]
    dotted_note_tokens = ["<NOTE_DOT_1>", "<NOTE_DOT_2>", "<NOTE_DOT_4>", "<NOTE_DOT_8>", "<NOTE_DOT_16>", "<NOTE_DOT_32>"]
    note_tokens = base_note_tokens + dotted_note_tokens
    
    # Value map for initialization (Unit: Quarter note = 1.0)
    dur_units = {
        "<NOTE_32>": 0.125,
        "<NOTE_DOT_32>": 0.1875,
        "<NOTE_16>": 0.25,
        "<NOTE_DOT_16>": 0.375,
        "<NOTE_8>": 0.5,
        "<NOTE_DOT_8>": 0.75,
        "<NOTE_4>": 1.0,
        "<NOTE_DOT_4>": 1.5,
        "<NOTE_2>": 2.0,
        "<NOTE_DOT_2>": 3.0,
        "<NOTE_1>": 4.0,
        "<NOTE_DOT_1>": 6.0,
    }
    
    
    bpm_tokens = [f"<BPM_{i}>" for i in range(256)]
    
    # Special tokens for SVS masking + score-block boundaries.
    # NOTE: this list is part of the released tokenizer vocabulary — do not
    # remove entries, or checkpoint embedding sizes will no longer match.
    special_tokens = [
        "<SVS_MASK>",
        # Score-block boundary markers.
        "<score_start>",
        "<score_end>",
        # Single-token rest marker for AP/SP word groups in the score
        # block. Replaces the multi-piece BPE encoding of "AP"/"SP"
        # text. Plain lyric block omits rests entirely (users do not
        # supply breath/silence locations).
        "<REST>",
    ]

    return pitch_tokens, note_tokens, bpm_tokens, dur_units, special_tokens


def estimate_svs_duration(svs_prompt: str, default_bpm: int = 120) -> float:
    """
    Estimate the audio duration from an SVS prompt.
    
    The estimation is based on:
    1. BPM (beats per minute) - extracted from <BPM_X> token
    2. Note durations - each note type has a duration in quarter note units
    
    Formula: 
        - unit_duration = 60 / BPM (seconds per quarter note)
        - total_duration = sum(dur_units[note] * unit_duration for note in notes)
    
    Args:
        svs_prompt: The SVS prompt string containing BPM and note tokens
        default_bpm: Default BPM to use if not found in prompt
        
    Returns:
        Estimated duration in seconds
    """
    import re
    
    # Get dur_units from the token maps (cached after first call)
    _, _, _, dur_units, _ = get_svs_token_maps()
    
    # Extract BPM using regex (fast)
    bpm = default_bpm
    bpm_match = re.search(r'<BPM_(\d+)>', svs_prompt)
    if bpm_match:
        bpm = int(bpm_match.group(1))
    
    # Calculate seconds per quarter note
    unit_duration = 60.0 / bpm
    
    # Sum up all note durations by directly looking up each token in dur_units
    total_units = 0.0
    for note_token, units in dur_units.items():
        # Count occurrences of this note token in the prompt
        count = svs_prompt.count(note_token)
        total_units += count * units
    
    # Calculate total duration in seconds
    total_duration = total_units * unit_duration
    
    return total_duration


def resize_token_embeddings_with_svs_init(model, tokenizer: PreTrainedTokenizer):
    """
    Resizes model embeddings and initializes SVS tokens with physical priors.
    """
    new_vocab_size = len(tokenizer)
    old_embeddings = model.base_lm.embed_tokens
    old_vocab_size, embedding_dim = old_embeddings.weight.shape

    # The score_lm_head resize is independent of the embed-table resize:
    # ``from_local`` auto-resizes ``embed_tokens`` to the checkpoint vocab
    # *before* this function is reached, but never touches
    # ``score_lm_head`` — so when the tokenizer + embed already match, we
    # would otherwise early-return with a stale score head and trip the
    # NLL out-of-range assert on the first training step. Resize the head
    # first (idempotent — only fires on shape mismatch), then handle the
    # embed table.
    score_head = getattr(model, "score_lm_head", None)
    if score_head is not None:
        old_out, old_in = score_head.weight.shape
        if old_out != new_vocab_size:
            new_head = nn.Linear(
                old_in, new_vocab_size, bias=score_head.bias is not None,
            )
            new_head = new_head.to(
                dtype=score_head.weight.dtype, device=score_head.weight.device,
            )
            new_head.weight.data.normal_(mean=0.0, std=0.02)
            copy_rows = min(old_out, new_vocab_size)
            new_head.weight.data[:copy_rows].copy_(score_head.weight.data[:copy_rows])
            if score_head.bias is not None:
                new_head.bias.data.zero_()
                new_head.bias.data[:copy_rows].copy_(score_head.bias.data[:copy_rows])
            model.score_lm_head = new_head
            print(
                f"Resized score_lm_head: {old_out} -> {new_vocab_size}"
            )

    if new_vocab_size == old_vocab_size:
        print(f"Vocab size match ({new_vocab_size}), skipping embed resize.")
        # Keep config in sync even when the embed layer didn't need a resize.
        model.config.lm_config.vocab_size = new_vocab_size
        model.base_lm.config.vocab_size = new_vocab_size
        return

    print(f"Resizing embeddings from {old_vocab_size} to {new_vocab_size}...")
    
    # 1. Create new embedding layer
    new_embeddings = nn.Embedding(
        new_vocab_size, 
        embedding_dim,
        dtype=old_embeddings.weight.dtype,
        device=old_embeddings.weight.device
    )
    
    # 2. Initialize with old weights
    # Standard vocab init (random for new parts first)
    new_embeddings.weight.data.normal_(mean=0.0, std=0.02)
    # Copy old weights
    new_embeddings.weight.data[:old_vocab_size, :] = old_embeddings.weight.data
    

    # 3. Smart Initialization for SVS Tokens
    pitch_tokens, note_tokens, bpm_tokens, dur_units, _ = get_svs_token_maps()
    
    # A. Pitch Initialization (Sinusoidal)
    print("Initializing Pitch Embeddings with Sinusoidal PE...")
    for token_str in pitch_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token_str)
        if token_id == tokenizer.unk_token_id:
            continue
            
        # Parse MIDI pitch from string "<P_60>"
        try:
            midi_pitch = int(token_str.split('_')[1][:-1])
        except:
            continue
            
        # Simplified approach: Use midi_pitch as 'position'
        
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        div_term = div_term.to(new_embeddings.weight.device)
        
        pe_sin = torch.sin(midi_pitch * div_term)
        pe_cos = torch.cos(midi_pitch * div_term)
        
        # Assign
        with torch.no_grad():
            new_embeddings.weight.data[token_id, 0::2] = pe_sin.to(new_embeddings.weight.dtype)
            if embedding_dim % 2 == 1:
                 new_embeddings.weight.data[token_id, 1::2] = pe_cos[:embedding_dim//2].to(new_embeddings.weight.dtype)
            else:
                 new_embeddings.weight.data[token_id, 1::2] = pe_cos.to(new_embeddings.weight.dtype)
                 

    # B. BPM Initialization (Sinusoidal)
    print("Initializing BPM Embeddings with Sinusoidal PE...")
    for token_str in bpm_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token_str)
        if token_id == tokenizer.unk_token_id:
            continue
            
        try:
            bpm_val = int(token_str.split('_')[1][:-1])
        except:
            continue
            
        # Treat BPM as a position (similar to Pitch)
        # However, to separate it from Pitch (0-127) and Note (Offset 1000+),
        # we can use a different offset or just trust the different dimensionality interpretation.
        # But to be safe and distinct, let's add an offset.
        # Pitch is 0-127. Let's put BPM at offset 2000.
        bpm_pos = bpm_val + 2000
        
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        div_term = div_term.to(new_embeddings.weight.device)
        
        pe_sin = torch.sin(bpm_pos * div_term)
        pe_cos = torch.cos(bpm_pos * div_term)
        
        with torch.no_grad():
            new_embeddings.weight.data[token_id, 0::2] = pe_sin.to(new_embeddings.weight.dtype)
            if embedding_dim % 2 == 1:
                 new_embeddings.weight.data[token_id, 1::2] = pe_cos[:embedding_dim//2].to(new_embeddings.weight.dtype)
            else:
                 new_embeddings.weight.data[token_id, 1::2] = pe_cos.to(new_embeddings.weight.dtype)

    # C. Duration Initialization (Relative Scale)
    print("Initializing Duration Embeddings with Relative Scaling...")
    
    for token_str in note_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token_str)
        if token_id == tokenizer.unk_token_id:
            continue
            
        # Calculate functional duration value:
        # Scale back to integer units (x8) so mathematical encoding remains similar to before
        # Add offset to separate from Pitch tokens (which use range 0-127)
        base_val = dur_units.get(token_str, 1.0)
        dur_val = (base_val * 8.0) + 1000
        
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        div_term = div_term.to(new_embeddings.weight.device)
        
        pe_sin = torch.sin(dur_val * div_term)
        pe_cos = torch.cos(dur_val * div_term)
        
        with torch.no_grad():
            new_embeddings.weight.data[token_id, 0::2] = pe_sin.to(new_embeddings.weight.dtype)
            if embedding_dim % 2 == 1:
                 new_embeddings.weight.data[token_id, 1::2] = pe_cos[:embedding_dim//2].to(new_embeddings.weight.dtype)
            else:
                 new_embeddings.weight.data[token_id, 1::2] = pe_cos.to(new_embeddings.weight.dtype)

    # 4. Replace model layers
    model.base_lm.embed_tokens = new_embeddings

    # 5. Update configs (score_lm_head was already resized at the top of this
    # function, before the embed-table early-return path).
    model.config.lm_config.vocab_size = new_vocab_size
    model.base_lm.config.vocab_size = new_vocab_size

    print("Embedding resize and initialization complete.")
