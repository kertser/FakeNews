from nicegui import ui

def detect():
    print(textInput.content)

def clear():
    pass

#%% --- Main Frame ---
ui.colors()

# Main Window:
with ui.card().classes('max-w-full no-wrap'):
    with ui.column().classes('max-w-full no-wrap'):
        ui.markdown('### Fake News Detector\nBy means of Emotional Analysis')
        #numbers = ui.table({'columnDefs': [{'field': 'numbers'}], 'rowData': []}).classes('max-h-40')
        ui.input(
            label='Input the sentence to be tested',
            placeholder='press ENTER to apply',
        ).classes('w-full').props('type=textarea')
        with ui.row():
            ui.button('Fake Detector', on_click=detect)
            ui.button('Clear Text', on_click=clear)
ui.html('<p>Alpha-Numerical, Mike Kertser, 2022, <strong>v0.01</strong></p>').classes('no-wrap')

if __name__ == "__main__":
    ui.run(title='Fake-News Tested', host='127.0.0.1', reload=False, show=True)
    #ui.run(title='Fake-News Tested', reload=True, show=True)