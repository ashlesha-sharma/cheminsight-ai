# ui.py

"""
A UI module featuring old money aesthetic.
This module includes reusable UI components styled with cream, burgundy, and gold accents,
and is built to be responsive, drawing inspiration from modern interfaces like Claude, Gemini, and ChatGPT.
"""

class OldMoneyButton:
    def __init__(self, text):
        self.text = text
        self.style = """
        background-color: burgundy;
        color: cream;
        border: 2px solid gold;
        padding: 10px 20px;
        border-radius: 5px;
        font-family: 'Times New Roman', serif;
        """

    def render(self):
        return f'<button style="{self.style}">{self.text}</button>'

class OldMoneyCard:
    def __init__(self, title, content):
        self.title = title
        self.content = content
        self.style = """
        background-color: cream;
        color: burgundy;
        border: 1px solid gold;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        """

    def render(self):
        return f'<div style="{self.style}"><h2>{self.title}</h2><p>{self.content}</p></div>'

class ResponsiveLayout:
    def __init__(self, components):
        self.components = components
        self.style = """
        display: flex;
        flex-wrap: wrap;
        justify-content: space-around;
        """

    def render(self):
        component_html = ''.join([component.render() for component in self.components])
        return f'<div style="{self.style}">{component_html}</div>'

# Example usage
if __name__ == '__main__':
    button = OldMoneyButton("Click Me")
    card = OldMoneyCard("Welcome", "This is an example card with an old money aesthetic.")
    layout = ResponsiveLayout([button, card])
    print(layout.render())
