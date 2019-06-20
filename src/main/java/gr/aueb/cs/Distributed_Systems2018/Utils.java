package gr.aueb.cs.Distributed_Systems2018;

import java.util.Random;

import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;

public class Utils {

	static Random r = new Random(1);
	
	static RandomGenerator randomGenerator;
	
	
	public static void init_JDKRandomGenerator() {
		randomGenerator = new JDKRandomGenerator();
		randomGenerator.setSeed(1);	
	}

	

	public static double generateRandomDouble(int rangeMin, int rangeMax) {

		double randomValue = rangeMin + (rangeMax - rangeMin) * r.nextDouble();

		return randomValue;
	}
	
	

}
