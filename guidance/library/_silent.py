import guidance

@guidance
def silent(self, **kwargs):
    open_text = f"""<||_html:<div style='display: inline-block; cursor: pointer; opacity: 0.5; width: 2px; margin-left: -2px;' onClick='this.nextSibling.style.display = "inline"; this.style.display = "none"'>&caron;</div><div style='display: none;'>_||>"""
    close_text = "<||_html:</div>_||>"
    return self.block(open_text=open_text, close_text=close_text)