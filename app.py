from tkinter import *
from chat import getResponse, chatBotName

backgroundGray = "#ABB2B9"
backgroundColor = "#17202A"
textColor = "#EAECEE"

font = "Helvetica 14"
fontBold = "Helvetica 13 bold"

class chatApp:
    
    def __init__(self):
        self.window = Tk()
        self._setup_main_window()
        
    def run(self):
        self.window.mainloop()
        
    def _setup_main_window(self):
        self.window.title("Chat")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=470, height=550, bg=backgroundColor)
        
        headLabel = Label(self.window, bg=backgroundColor, fg=textColor,
                           text="Qassim Universty", font=fontBold, pady=10)
        headLabel.place(relwidth=1)
        
        line = Label(self.window, width=450, bg=backgroundGray)
        line.place(relwidth=1, rely=0.07, relheight=0.012)
        
        self.textwidg = Text(self.window, width=20, height=2, bg=backgroundColor, fg=textColor,
                                font=font, padx=5, pady=5)
        self.textwidg.place(relheight=0.745, relwidth=1, rely=0.08)
        self.textwidg.configure(cursor="arrow", state=DISABLED)
        
        scrollBar = Scrollbar(self.textwidg)
        scrollBar.place(relheight=1, relx=0.974)
        scrollBar.configure(command=self.textwidg.yview)
        
        bottomLabel = Label(self.window, bg=backgroundGray, height=80)
        bottomLabel.place(relwidth=1, rely=0.825)
        
        self.message_enter = Entry(bottomLabel, bg="#2C3E50", fg=textColor, font=font)
        self.message_enter.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.011)
        self.message_enter.focus()
        self.message_enter.bind("<Return>", self._on_enter_pressed)
        
        sendButton = Button(bottomLabel, text="Send", font=fontBold, width=20, bg=backgroundGray,command=lambda: self._on_enter_pressed(None))
        sendButton.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)
     
    def _on_enter_pressed(self, event):
        message = self.message_enter.get()
        self._insert_message(message, "You")
        
    def _insert_message(self, message, sender):
        if not message:
            return
        
        self.message_enter.delete(0, END)
        message1 = f"{sender}: {message}\n\n"
        self.textwidg.configure(state=NORMAL)
        self.textwidg.insert(END, message1)
        self.textwidg.configure(state=DISABLED)
        
        message2 = f"{chatBotName}: {getResponse(message)}\n\n"
        self.textwidg.configure(state=NORMAL)
        self.textwidg.insert(END, message2)
        self.textwidg.configure(state=DISABLED)
        
        self.textwidg.see(END)
             
        
if __name__ == "__main__":
    app = chatApp()
    app.run()