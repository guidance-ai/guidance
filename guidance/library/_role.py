import guidance

@guidance(model=guidance.models.Chat)
def role(self, role_name, text=None, **kwargs):
    open_text = f"<||_html:<div style='display: flex; border-bottom: 1px solid rgba(127, 127, 127, 0.2); align-items: center;'><div style='flex: 0 0 80px; opacity: 0.5;'>{role_name.lower()}</div><div style='flex-grow: 1; padding: 5px; padding-top: 10px; padding-bottom: 10px; margin-top: 0px; white-space: pre-wrap; margin-bottom: 0px;'>_||>"
    open_text += "<||_#NODISP_||>" + self.get_role_start(role_name, **kwargs) + "<||_/NODISP_||>"
    close_text = "<||_html:</div></div>_||>" + "<||_#NODISP_||>" + self.get_role_end(role_name) + "<||_/NODISP_||>"
    if text is None:
        return self.block(open_text=open_text, close_text=close_text)
    else:
        return self.append(open_text + text + close_text)

@guidance(model=guidance.models.Chat)
def system(self, text=None, **kwargs):
    return self.role("system", text, **kwargs)

@guidance(model=guidance.models.Chat)
def user(self, text=None, **kwargs):
    return self.role("user", text, **kwargs)

@guidance(model=guidance.models.Chat)
def assistant(self, text=None, **kwargs):
    return self.role("assistant", text, **kwargs)

@guidance(model=guidance.models.Chat)
def function(self, text=None, **kwargs):
    return self.role("function", text, **kwargs)