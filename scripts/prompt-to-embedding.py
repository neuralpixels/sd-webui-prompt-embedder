import gradio as gr
from modules import script_callbacks, shared, sd_hijack
from modules.shared import cmd_opts
import torch, os
from modules.textual_inversion.textual_inversion import Embedding
import collections, math, random

EXTENSION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EMB_SAVE_EXT = '.pt'  # '.bin'


# -------------------------------------------------------------------------------

def get_data():
    loaded_embs = collections.OrderedDict(
        sorted(
            sd_hijack.model_hijack.embedding_db.word_embeddings.items(),
            key=lambda x: str(x[0]).lower()
        )
    )

    embedder = shared.sd_model.cond_stage_model.wrapped
    if embedder.__class__.__name__ == 'FrozenCLIPEmbedder':  # SD1.x detected
        tokenizer = embedder.tokenizer
        internal_embs = embedder.transformer.text_model.embeddings.token_embedding.wrapped.weight

    elif embedder.__class__.__name__ == 'FrozenOpenCLIPEmbedder':  # SD2.0 detected
        from modules.sd_hijack_open_clip import tokenizer as open_clip_tokenizer
        tokenizer = open_clip_tokenizer
        internal_embs = embedder.model.token_embedding.wrapped.weight

    else:
        tokenizer = None
        internal_embs = None

    return tokenizer, internal_embs, loaded_embs  # return these useful references


# -------------------------------------------------------------------------------

def text_to_emb_ids(text, tokenizer):
    text = text.lower()

    if tokenizer.__class__.__name__ == 'CLIPTokenizer':  # SD1.x detected
        emb_ids = tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]

    elif tokenizer.__class__.__name__ == 'SimpleTokenizer':  # SD2.0 detected
        emb_ids = tokenizer.encode(text)

    else:
        emb_ids = None

    return emb_ids  # return list of embedding IDs for text


# -------------------------------------------------------------------------------

def emb_id_to_name(emb_id, tokenizer):
    emb_name_utf8 = tokenizer.decoder.get(emb_id)

    if emb_name_utf8 != None:
        byte_array_utf8 = bytearray([tokenizer.byte_decoder[c] for c in emb_name_utf8])
        emb_name = byte_array_utf8.decode("utf-8", errors='backslashreplace')
    else:
        emb_name = '!Unknown ID!'

    return emb_name  # return embedding name for embedding ID


# -------------------------------------------------------------------------------

def get_embedding_info(text):
    text = text.lower()

    tokenizer, internal_embs, loaded_embs = get_data()

    loaded_emb = loaded_embs.get(text, None)

    if loaded_emb == None:
        for k in loaded_embs.keys():
            if text == k.lower():
                loaded_emb = loaded_embs.get(k, None)
                break

    if loaded_emb != None:
        emb_name = loaded_emb.name
        emb_id = '[' + loaded_emb.checksum() + ']'  # emb_id is string for loaded embeddings
        emb_vec = loaded_emb.vec
        return emb_name, emb_id, emb_vec, loaded_emb  # also return loaded_emb reference

    # support for #nnnnn format
    val = None
    if text.startswith('#'):
        try:
            val = int(text[1:])
            if (val < 0) or (val >= internal_embs.shape[0]): val = None
        except:
            val = None

    # obtain internal embedding ID
    if val != None:
        emb_id = val
    else:
        emb_ids = text_to_emb_ids(text, tokenizer)
        if len(emb_ids) == 0: return None, None, None, None
        emb_id = emb_ids[0]  # emb_id is int for internal embeddings

    emb_name = emb_id_to_name(emb_id, tokenizer)
    emb_vec = internal_embs[emb_id].unsqueeze(0)

    return emb_name, emb_id, emb_vec, None  # return embedding name, ID, vector


# -------------------------------------------------------------------------------
def do_make_it(*args):
    prompt = args[0].strip().lower()
    save_name = args[1]
    num_tokens = args[2]
    do_overwrite = args[3]
    method = args[4]
    filetype = args[5]



    results = []
    details = []
    details_end = []

    if save_name == '':
        return 'Filename is empty', ''
    save_filename_st = os.path.join(cmd_opts.embeddings_dir, f"{save_name}.safetensors")
    save_filename_pt = os.path.join(cmd_opts.embeddings_dir, f"{save_name}.pt")
    save_filename = os.path.join(cmd_opts.embeddings_dir, f"{save_name}.{filetype}")

    # check if with both extensions
    file_exists = os.path.exists(save_filename_st) or os.path.exists(save_filename_pt)
    if file_exists:
        if not do_overwrite:
            return f'Embedding already exists, use "overwrite" option to overwrite it', ''
        else:
            results.append('File already exists, overwrite overwriting it')
            # remove other if filename is different
            if os.path.exists(save_filename_st) and (save_filename_st != save_filename):
                os.remove(save_filename_st)
            if os.path.exists(save_filename_pt) and (save_filename_pt != save_filename):
                os.remove(save_filename_pt)

    tokenizer, internal_embs, loaded_embs = get_data()
    # list of tokens, they normally get a # prefix
    token_ids = text_to_emb_ids(prompt, tokenizer)

    details.append(f'Your prompt generated {len(token_ids)} tokens')

    anything_saved = False

    # calculate mixed embedding in tot_vec
    out_vec = None
    out_vec_list = []
    cur_index = 0
    for k in range(len(token_ids)):
        name = f'#{token_ids[k]}'.strip().lower()

        mixval = 1  # multiplier for this embedding
        if (name == '') or (mixval == 0):
            continue

        emb_name, emb_id, emb_vec, loaded_emb = get_embedding_info(name)
        token_vec = emb_vec.to(device='cpu', dtype=torch.float32)

        token_vectors = token_vec.shape[0]
        # create our tot_vec if it is none
        if out_vec is None:
            # vectors are shape of (num_tokens, 768/1024) depending on clip version
            out_vec = torch.zeros(num_tokens, token_vec.shape[1]).to(device='cpu', dtype=torch.float32)

        details.append(f' - {emb_name} ({str(emb_id)}) - {token_vectors} token{"" if token_vectors == 1 else "s"} -> l:{len(out_vec_list)}, p:{cur_index}')

        # walk each vector one by one
        for v in range(token_vec.shape[0]):
            out_vec[cur_index] += token_vec[v]
            # increase index
            cur_index += 1
            if cur_index >= num_tokens:
                # reached the end, reset back to the start
                cur_index = 0
                out_vec_list.append(out_vec.unsqueeze(0))
                out_vec = None

    # add remaining vectors
    if out_vec is not None:
        out_vec_list.append(out_vec.unsqueeze(0))

    results.append(f"Total embedding stacks: {len(out_vec_list)}")
    stacked_vec = torch.cat(out_vec_list, dim=0)
    if method == 'mean':
        tot_vec = torch.mean(stacked_vec, dim=0)
    else:
        # assume sum
        tot_vec = torch.sum(stacked_vec, dim=0)

    if tot_vec.shape[0] > 0:

        results.append('Final embedding size: ' + str(tot_vec.shape[0]) + ' x ' + str(tot_vec.shape[1]))

        new_emb = Embedding(tot_vec, save_name)
        new_emb.step = 0

        try:
            new_emb.save(save_filename)
            results.append('Saved "' + save_filename + '"')
            anything_saved = True

        except:
            results.append('🛑 Error saving "' + save_filename + '" (filename might be invalid)')

    # ------------- end batch loop

    if anything_saved:

        results.append('Reloading all embeddings')
        try:  # new way
            sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload=True)
        except:  # old way
            sd_hijack.model_hijack.embedding_db.dir_mtime = 0
            sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()

        results.append('Your embedding is ready to use!')

    return '\n'.join(results), '\n'.join(details)


# -------------------------------------------------------------------------------

def get_about_html():
    about_path = os.path.join(EXTENSION_ROOT, 'assets', 'about.html')
    # load the ocntents
    with open(about_path, 'r') as f:
        about_html = f.read()
    return about_html


def add_tab():
    with gr.Blocks(analytics_enabled=False) as ui:
        with gr.Tabs():
            with gr.Row():
                with gr.Column(variant='panel'):
                    prompt_input = gr.Textbox(
                        label="Your Prompt Here",
                        lines=4,
                        placeholder="Enter a prompt here. It can be as long as you want. It can include other embeddings. It cannot handle weight modifiers yet like (this). Then click 'Make it!'",
                    )

                    with gr.Row():
                        save_name_input = gr.Textbox(label="Filename", lines=1,
                                                     placeholder='Embedding Name (without extension)')
                        num_tokens_slider = gr.Slider(
                            minimum=1,
                            maximum=74,
                            value=1,
                            step=1,
                            label="Num Tokens",
                            info="The number of tokens your embedding will consume."
                        )
                    with gr.Row():
                        method_selector = gr.Dropdown(
                            label="Method",
                            choices=['mean', 'sum'],
                            value='sum',
                            variant='secondary',
                            info="How to combine the stacks of tokens into a single embedding."
                        )
                        filetype_selector = gr.Dropdown(
                            label="Filetype",
                            choices=['pt', 'safetensors'],
                            value='pt',
                            variant='secondary',
                            info="The file format to save the embedding in."
                        )
                    with gr.Row():
                        make_it_button = gr.Button(value="Make It!", variant="primary")
                        overwrite_checkbox = gr.Checkbox(value=False, label="Overwrite if exists")
                    with gr.Row():
                        save_result_output = gr.Textbox(label="Output", lines=8)
                        save_result_details = gr.Textbox(label="Details", lines=8)

                with gr.Column(variant='panel'):
                    html_output = gr.HTML(
                        label="About this thing",
                        show_label=False,
                        elem_id="prompt-embedder-about",
                        value=get_about_html
                    )

            make_it_button.click(
                fn=do_make_it,
                inputs=[
                    prompt_input, save_name_input, num_tokens_slider, overwrite_checkbox,
                    method_selector, filetype_selector
                ],
                outputs=[save_result_output, save_result_details]
            )

    return [(ui, "Prompt Embedder", "prompt_to_embedding")]


script_callbacks.on_ui_tabs(add_tab)
