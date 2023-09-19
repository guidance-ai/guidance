import guidance

@guidance
def silent(self, **kwargs):
    open_text = f"<||_html:<div style='display: none;'>_||>"
    close_text = "<||_html:</div>_||>"
    return self.block(open_text=open_text, close_text=close_text)