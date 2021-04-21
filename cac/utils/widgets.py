from ipywidgets import Text, Layout, Button, Dropdown, IntText


def define_inttext(value=None, placeholder='', description='', layout=Layout(),
                   style={}):
    text = IntText(
        value=value,
        placeholder=placeholder,
        description=description,
        disabled=False,
        layout=layout,
        style=style
    )
    return text


def define_text(value='', placeholder='', description='', layout=Layout(),
                style={}):
    text = Text(
        value=value,
        placeholder=placeholder,
        description=description,
        disabled=False,
        layout=layout,
        style=style
    )
    return text


def define_button(desc, layout=Layout(), style={}):
    button = Button(description=desc, layout=layout, style=style)
    return button


def define_dropdown(options, default, desc, layout=Layout(), style={}):
    dropdown = Dropdown(
        options=options,
        value=default,
        description=desc,
        disabled=False,
        layout=layout,
        style=style
    )
    return dropdown
