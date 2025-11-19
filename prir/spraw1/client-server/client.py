import socket
import threading
import tkinter as tk
from tkinter import scrolledtext, messagebox
import platform
import getpass

def get_client_info():
    info = {
        "username": getpass.getuser(),
        "hostname": platform.node(),
        "os": platform.system(),
    }
    return info


class ClientGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Klient TCP – GUI")
        self.sock = None
        self.connected = False

        tk.Label(root, text="Adres serwera:").pack()
        self.host_entry = tk.Entry(root)
        self.host_entry.insert(0, "127.0.0.1")
        self.host_entry.pack()

        tk.Label(root, text="Port:").pack()
        self.port_entry = tk.Entry(root)
        self.port_entry.insert(0, "54321")
        self.port_entry.pack()

        self.connect_button = tk.Button(root, text="Połącz", command=self.connect_to_server)
        self.connect_button.pack(pady=5)

        self.disconnect_button = tk.Button(root, text="Rozłącz", command=self.disconnect, state=tk.DISABLED)
        self.disconnect_button.pack(pady=5)

        tk.Label(root, text="Odebrane dane:").pack()
        self.log_area = scrolledtext.ScrolledText(root, width=60, height=20)
        self.log_area.pack()

        tk.Label(root, text="Wiadomość:").pack()
        self.msg_entry = tk.Entry(root, width=50)
        self.msg_entry.pack()

        self.send_button = tk.Button(root, text="Wyślij", command=self.send_message, state=tk.DISABLED)
        self.send_button.pack(pady=5)

    def log(self, msg):
        self.log_area.insert(tk.END, msg + "\n")
        self.log_area.see(tk.END)

    def connect_to_server(self):
        host = self.host_entry.get()
        port = int(self.port_entry.get())

        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(5)
            self.sock.connect((host, port))
            self.sock.settimeout(None)
            import json
            self.sock.send(json.dumps(get_client_info()).encode())
            self.connected = True

            self.connect_button.config(state=tk.DISABLED)
            self.disconnect_button.config(state=tk.NORMAL)
            self.send_button.config(state=tk.NORMAL)

            self.log("Połączono z serwerem.")

            threading.Thread(target=self.receive_messages, daemon=True).start()
        except socket.timeout:
            self.log("⛔ Połączenie nieudane – serwer nie odpowiada (timeout 5s).")
            messagebox.showerror("Błąd", "Nie udało się połączyć z serwerem w 5 sekund.")
            self.sock.close()
            self.sock = None
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie udało się połączyć: {e}")

    def disconnect(self):
        if self.sock:
            try:
                self.sock.close()
            except:
                pass

        self.connected = False
        self.connect_button.config(state=tk.NORMAL)
        self.disconnect_button.config(state=tk.DISABLED)
        self.send_button.config(state=tk.DISABLED)

        self.log("Rozłączono z serwerem.")

    def receive_messages(self):
        while self.connected:
            try:
                data = self.sock.recv(1024)
                if not data:
                    break

                msg = data.decode()

                if msg == "SERVER_SHUTDOWN":
                    self.log("⚠️ Serwer został wyłączony.")
                    break

                self.log(msg)

            except:
                break

        self.disconnect()

    def send_message(self):
        msg = self.msg_entry.get()
        if msg and self.connected:
            try:
                self.sock.send(msg.encode())
                self.log("Ty → " + msg)
                self.msg_entry.delete(0, tk.END)
            except:
                self.log("Błąd wysyłania.")
        else:
            messagebox.showwarning("Uwaga", "Nie połączono z serwerem.")


if __name__ == "__main__":
    root = tk.Tk()
    app = ClientGUI(root)
    root.mainloop()
