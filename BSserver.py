import asyncio
import socket

async def handle_client(reader, writer):
    client_address = writer.get_extra_info('peername')
    print(f'Connected to client at {client_address[0]}:{client_address[1]}')

    while True:
        try:
            data = await reader.read(1024)
            message = data.decode().strip()
            if not message:
                print("No message received from client.")
                break
            print(f"Received message from client: {message}")
            # 在这里添加逻辑来处理客户端发送的消息，然后构造响应
            # 这里只是一个示例，可以根据需要进行修改
            
            response = "Hello from server!"
            writer.write(response.encode())
            
            await writer.drain()
        except Exception as e:
            print(f"An error occurred: {e}")
            break

    writer.close()
    await writer.wait_closed()

async def main():
    server_ip = '192.168.1.108'  # 服务器的IP地址
    server_port = 22223  # 服务器的端口号

    server = await asyncio.start_server(
        handle_client, server_ip, server_port)

    print(f'Server is listening on {server_ip}:{server_port}')

    async with server:
        await server.serve_forever()

asyncio.run(main())
