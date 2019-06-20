package gr.aueb.cs.Distributed_Systems2018;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.Socket;
import java.net.UnknownHostException;

public class Client {

	Socket clientSocket;
	ObjectInputStream in;
	ObjectOutputStream out;

	String ID = "Client";

	public static void main(String[] args) {

		Client client = new Client();

		client.startClient();

	}

	public void startClient() {

		// connect with Master
		try {
			clientSocket = new Socket(Constants.MASTER_IP, Constants.MASTER_PORT);

			out = new ObjectOutputStream(clientSocket.getOutputStream());
			in = new ObjectInputStream(clientSocket.getInputStream());

			String message = (String) in.readObject();
			System.out.println("Server says: " + message);

			out.writeObject(ID);
			out.flush();

			while (true) {
				double error = (Double) in.readObject();

				System.out.println("Client received error: " + error);
			}
		} catch (UnknownHostException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}

	}

}
