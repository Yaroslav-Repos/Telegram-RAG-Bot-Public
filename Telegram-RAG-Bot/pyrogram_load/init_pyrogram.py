from pyrogram import Client

API_ID = ***************
API_HASH = "**************"

for i in range(5):
    session_name = f"test_user_{i}"
    print(f"[*] Initializing session: {session_name}")

    app = Client(session_name, api_id=API_ID, api_hash=API_HASH)

    app.start() 
    app.stop()

    print(f"[+] Session {session_name} created\n")
