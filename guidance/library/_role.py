import guidance

@guidance(model=guidance.models.ChatLM)
def role(self, role_name, text=None):
    open_text = f"<||_html:<div style='display: flex; border-bottom: 1px solid rgba(127, 127, 127, 0.2); align-items: center;'><div style='flex: 0 0 80px; opacity: 0.5;'>{role_name.lower()}</div><div style='flex-grow: 1; padding: 5px; padding-top: 10px; padding-bottom: 10px; margin-top: 0px; white-space: pre-wrap; margin-bottom: 0px;'>_||>"
    open_text += "<||_#NODISP_||>" + self.get_role_start(role_name) + "<||_/NODISP_||>"
    close_text = "<||_html:</div></div>_||>" + "<||_#NODISP_||>" + self.get_role_end(role_name) + "<||_/NODISP_||>"
    if text is None:
        return self.block(open_text, close_text)
    else:
        return self + open_text + text + close_text

@guidance(model=guidance.models.ChatLM)
def system(self, text=None):
    return self.role("system", text)

@guidance(model=guidance.models.ChatLM)
def user(self, text=None):
    return self.role("user", text)

@guidance(model=guidance.models.ChatLM)
def assistant(self, text=None):
    return self.role("assistant", text)

@guidance(model=guidance.models.ChatLM)
def function(self, text=None):
    return self.role("function", text)