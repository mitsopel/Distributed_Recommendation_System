package gr.aueb.cs.Distributed_Systems2018;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.math.BigInteger;
import java.net.Socket;

public class Connection {

	String name; // client or map worker
	Socket socket;
	ObjectOutputStream out;
	ObjectInputStream in;

	public Connection(String name, Socket socket, ObjectOutputStream out, ObjectInputStream in) {
		super();
		this.name = name;
		this.socket = socket;
		this.out = out;
		this.in = in;
	}

}
