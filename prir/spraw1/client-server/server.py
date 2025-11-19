import socket
import threading
import tkinter as tk
from tkinter import scrolledtext, messagebox


class ServerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Serwer TCP – GUI")
        self.server_socket = None
        self.running = False
        self.clients = {}

        ip_frame = tk.Frame(root)
        ip_frame.pack(pady=5)

        tk.Label(ip_frame, text="IP serwera:").grid(row=0, column=0)
        self.server_ip_entry = tk.Entry(ip_frame)
        self.server_ip_entry.insert(0, f"{socket.gethostbyname(socket.gethostname())}")
        self.server_ip_entry.grid(row=0, column=1)

        tk.Label(ip_frame, text="Port:").grid(row=1, column=0)
        self.server_port_entry = tk.Entry(ip_frame)
        self.server_port_entry.insert(0, "54321")
        self.server_port_entry.grid(row=1, column=1)

        self.start_button = tk.Button(root, text="Start serwera", command=self.start_server)
        self.start_button.pack(pady=5)

        self.stop_button = tk.Button(root, text="Stop serwera", command=self.stop_server, state=tk.DISABLED)
        self.stop_button.pack(pady=5)

        tk.Label(root, text="Aktywni klienci:").pack()
        self.client_list = tk.Listbox(root, height=6, width=40)
        self.client_list.pack(padx=10, pady=5)

        tk.Label(root, text="Wiadomość do klienta:").pack()
        self.server_msg_entry = tk.Entry(root, width=40)
        self.server_msg_entry.pack(pady=5)

        self.send_to_client_btn = tk.Button(root, text="Wyślij", command=self.send_to_selected_client)
        self.send_to_client_btn.pack(pady=5)

        tk.Label(root, text="Log serwera:").pack()
        self.log_area = scrolledtext.ScrolledText(root, width=60, height=20)
        self.log_area.pack(padx=10, pady=5)

    def log(self, message):
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.see(tk.END)

    def start_server(self):
        HOST = self.server_ip_entry.get()
        PORT = int(self.server_port_entry.get())

        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            self.server_socket.bind((HOST, PORT))
            self.server_socket.listen(5)
            self.running = True

            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)

            self.log(f"Serwer uruchomiony na {HOST}:{PORT}")

            threading.Thread(target=self.accept_clients, daemon=True).start()

        except Exception as e:
            messagebox.showerror("Błąd", f"Nie można uruchomić serwera: {e}")

    def stop_server(self):
        self.running = False
        self.log("Zatrzymywanie serwera...")

        for conn in list(self.clients.keys()):
            try:
                conn.send("SERVER_SHUTDOWN".encode())
                conn.shutdown(socket.SHUT_RDWR)
            except:
                pass
            try:
                conn.close()
            except:
                pass

        self.clients.clear()
        self.client_list.delete(0, tk.END)

        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass

        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

        self.log("Serwer zatrzymany. Port dostępny.")

    def accept_clients(self):
        while self.running:
            try:
                conn, addr = self.server_socket.accept()
            except OSError:
                break
            data = conn.recv(1024)

            import json
            client_info = json.loads(data.decode())
            self.clients[conn] = {
                "addr": addr,
                "username": client_info.get("username", "Unknown"),
                "hostname": client_info.get("hostname", "Unknown"),
                "os": client_info.get("os", "Unknown"),
            }
            self.client_list.insert(tk.END,
                                    f"{addr[0]}:{addr[1]} | {self.clients[conn]['username']}@{self.clients[conn]['hostname']} ({self.clients[conn]['os']})")
            self.log(f"[+] Połączono: {addr}")

            threading.Thread(target=self.handle_client, args=(conn,), daemon=True).start()

    def handle_client(self, conn):
        addr = self.clients[conn]

        while self.running:
            try:
                data = conn.recv(1024)
            except:
                break

            if not data:
                break

            msg = data.decode().strip()
            self.log(f"[{addr['addr'][0]}:{addr['addr'][1]} | {addr['username']}@{addr['hostname']}({addr['os']})] > {msg}")

        conn.close()
        if conn in self.clients:
            del self.clients[conn]

        self.client_list.delete(0, tk.END)
        for c in self.clients.values():
            addr = c.get("addr")
            username = c.get("username", "???")
            hostname = c.get("hostname", "???")
            os_name = c.get("os", "???")
            self.client_list.insert(tk.END, f"{addr[0]}:{addr[1]} | {username}@{hostname} ({os_name})")

        self.log(f"[-] Rozłączono: {addr}")

    def send_to_selected_client(self):
        selection = self.client_list.curselection()
        if not selection:
            messagebox.showwarning("Uwaga", "Wybierz klienta z listy.")
            return

        msg = self.server_msg_entry.get()
        if not msg:
            messagebox.showwarning("Uwaga", "Wiadomość nie może być pusta.")
            return

        index = selection[0]
        target_conn = list(self.clients.keys())[index]

        try:
            addr = self.clients[target_conn]
            target_conn.send(f"[SERVER]: {msg}".encode())
            self.log(f"[SERVER → {addr['addr'][0]}:{addr['addr'][1]} | {addr['username']}@{addr['hostname']}({addr['os']})] {msg}")
        except:
            self.log("Błąd wysyłania wiadomości.")

        self.server_msg_entry.delete(0, tk.END)


if __name__ == "__main__":
    root = tk.Tk()
    app = ServerGUI(root)
    root.mainloop()
