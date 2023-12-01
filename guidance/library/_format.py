from ._block import block

def monospace():
    return block(opener="<||_html:<span style='font-family: Menlo, Monaco, monospace; font-size: 13px;'>_||>", closer="<||_html:</span>_||>")

