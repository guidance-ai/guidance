import random

class Random():
    def __init__(self, items, k=1):
        ''' Create a selector that chooses a random set of instances.

        Parameters
        ----------
        k : int
            The number of instances to choose.
        '''
        self.items = items
        self.k = k
    
    def __call__(self):
        ''' Selects a random set of instances.
        '''
        return random.choice(self.items, k=self.k)
    
class NGramOverlap():
    def __init__(self, items, k=1):
        ''' Create a selector that chooses a random set of instances.

        Parameters
        ----------
        k : int
            The number of instances to choose.
        '''
        # TODO: Implement this.
        # It would be nice just use let the LangChain option cover this, but they have some odd dependencies on an example template.
        raise NotImplementedError("NGramOverlap is not implemented yet.")
    
    def __call__(self):
        ''' Selects a random set of instances.
        '''
        return self.items

def _word_tokenizer(text):
    ''' Tokenizes a string by white space.
    '''
    return text.split()

class TokenLimit():
    
    def __init__(self, items, max_tokens=20):
        ''' Create a selector that limits the number of tokens in a list of items.

        Parameters
        ----------
        items : list
            A list of items to select from.
        max_tokens : int
            The maximum number of tokens to allow.
        '''

        self.items = items
        self.max_tokens = max_tokens
    
    def __call__(self, template_context=None):
        ''' Filters a list of items to a maximum number of tokens.

        Parameters
        ----------
        template_context : dict (optional)
            A dictionary of template context variables to use for token counting.
        
        Returns
        -------
        A list of items that fit within the token limit.
        '''
        
        if template_context is not None and "@tokenizer" in template_context:
            token_encoder = template_context["@tokenizer"].encode
        else:
            token_encoder = _word_tokenizer
        total_length = 0
        out = []
        for item in self.items:
            if template_context is not None and "@block_text" in template_context:
                context_new = template_context["@block_text"]
                if isinstance(item, dict):
                    for k in item:
                        context_new = context_new.replace("{{this."+k+"}}", item[k])
                else:
                    context_new = context_new.replace("{{this}}", item)
            else:
                context_new = " ".join([item[k] for k in item])
            new_length = len(token_encoder(context_new))
            if total_length + new_length <= self.max_tokens:
                total_length += new_length
                out.append(item)
        return out
    

class LangChain():
    def __init__(self, selector):
        ''' Create a selector from a LangChain ExampleSelector object.

        Parameters
        ----------
        selector : ExampleSelector
            A LangChain ExampleSelector object.
        
        Returns
        -------
        A selector that selects examples using a LangChain ExampleSelector object.
        '''
        self.selector = selector
    
    def __call__(self, **kwargs):
        ''' Select examples using a LangChain ExampleSelector object.

        Note that we use keyword arguments here instead of a single dictionary.
        '''
        out = self.selector.select_examples(kwargs)
        return out