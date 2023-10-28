from ._block import block

def silent():
    open_text = f"""<||_html:<div style='display: inline-block; cursor: pointer; opacity: 0.5; width: 2px; margin-left: -2px;' onClick='this.nextSibling.style.display = "inline"; this.style.display = "none"'>&caron;</div><div style='display: none;'>_||>"""
    close_text = "<||_html:</div>_||>"
    return block(opener=open_text, closer=close_text)