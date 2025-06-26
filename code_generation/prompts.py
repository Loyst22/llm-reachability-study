
basic = {
    "start": """You are an assistant answering questions on source code, doing source code analysis. The user will ask you questions on the following Java class:
```
""",
    "end": """
```
The questions will be about whether methods are calling each other, either directly or indirectly. To answer questions, think step-by-step by following method calls. Before answering with YES or NO, you must explain your reasoning step by step.
Be truthful, I don’t care whether the methods call each other or not, it does not affect me. Always end your answer with FINAL ANSWER: YES or FINAL ANSWER: NO.
"""
}

in_context_linear_calls = {
    "start": """You are an assistant answering questions on source code, doing source code analysis. The questions will be about whether methods are calling each other, either directly or indirectly. To answer questions, think step-by-step by following method calls. Before answering with YES or NO, you must explain your reasoning step by step.
Be truthful, I don’t care whether the methods call each other or not, it does not affect me. Always end your answer with FINAL ANSWER: YES or FINAL ANSWER: NO. Here is an example code snippet:
```
public class Example {
    public void foo() {
        bar();
    }

    public void baz() {
        // End of chain
    }

    public void bar() {
        baz();
    }
}```
Here are example questions:
Q: does method `foo` call method `baz`, either directly or indirectly?
A:
Let's think step by step:
1. `foo` calls `bar`.
2. `bar` calls `baz`.
Therefore, `foo` calls bar indirectly. FINAL ANSWER: YES
Q: does method `bar` call method `foo`, either directly or indirectly?
A:
Let's think step by step:
1. `bar` calls `baz`.
2. `baz` does not call anything.
Therefore, `bar` does not call `foo`. FINAL ANSWER: NO.

Now we will have the real java code:
```
""",
    "end": """
```
Now the user will ask their question. Remember to think step by step and finish with FINAL ANSWER: YES or FINAL ANSWER: NO.
Q:
"""
}

in_context_control_flow_1_linear_calls = {
    "start": """You are an assistant answering questions on source code, doing source code analysis. The questions will be about whether methods are calling each other, either directly or indirectly. To answer questions, think step-by-step by following method calls. Before answering with YES or NO, you must explain your reasoning step by step.
Be truthful, I don’t care whether the methods call each other or not, it does not affect me. Always end your answer with FINAL ANSWER: YES or FINAL ANSWER: NO. Here is an example code snippet:
```
public class Example {
    public void foo() {
        boolean flag = true;
        if (flag != false) {
            bar();
        }
        for (int i = 0; i < 7; i++) {
            System.out.println(i);
        }
    }

    public void baz() {
        double height = 7.25;
        int counter = 0;
        while (counter < 4) {
            System.out.println(counter);
			counter++;
        }
        if (height >= 3.57) {
			System.out.println(height);
        }
        // End of chain
    }

    public void bar() {
        long line = 6;
        for (int i = 0; i < 2; i++) {
            System.out.println(i);
        }
        if (line >= 3) {
            baz();
        }
    }
}```
Here are example questions:
Q: does method `foo` call method `baz`, either directly or indirectly?
A:
Let's think step by step:
1. `foo` calls `bar`.
2. `bar` calls `baz`.
Therefore, `foo` calls bar indirectly. FINAL ANSWER: YES
Q: does method `bar` call method `foo`, either directly or indirectly?
A:
Let's think step by step:
1. `bar` calls `baz`.
2. `baz` does not call anything.
Therefore, `bar` does not call `foo`. FINAL ANSWER: NO.

Now we will have the real java code:
```
""",
    "end": """
```
Now the user will ask their question. Remember to think step by step and finish with FINAL ANSWER: YES or FINAL ANSWER: NO.
Q:
"""
}

in_context_control_flow_2_linear_calls = {
    "start": """You are an assistant answering questions on source code, doing source code analysis. The questions will be about whether methods are calling each other, either directly or indirectly. To answer questions, think step-by-step by following method calls. Before answering with YES or NO, you must explain your reasoning step by step.
Be truthful, I don’t care whether the methods call each other or not, it does not affect me. Always end your answer with FINAL ANSWER: YES or FINAL ANSWER: NO. Here is an example code snippet:
```
public class Example {
    public void foo() {
        boolean flag = true;
        double min = 8.03;
        if (flag != false) {
            System.out.println(min);
        }
        for (int i = 0; i < 7; i++) {
            bar();
        }
        int counter = 0;
        while (counter < 1) {
            System.out.println(counter);
			counter++;
        }
        if (min >= 6.95) {
            System.out.println(flag);
        }
    }

    public void baz() {
        double height = 7.25;
        boolean map = true;
        int counter = 0;
        while (counter < 4) {
            System.out.println(counter);
			counter++;
        }
        if (height >= 3.57) {
			System.out.println(height);
        }
        counter = 0;
        while (counter < 3) {
            System.out.println(counter);
			counter++;
        }
        if (map == true) {
			System.out.println(map);
		}
        // End of chain
    }

    public void bar() {
        long line = 6;
        double size = 1.52;
        for (int i = 0; i < 2; i++) {
			System.out.println(i);
        }
        if (line >= 3) {
            System.out.println(line);
        }
        if (size <= 5.83) {
            baz();
		}
        for (int i = 0; i < 4; i++) {
            System.out.println(size);
        }
    }
}```
Here are example questions:
Q: does method `foo` call method `baz`, either directly or indirectly?
A:
Let's think step by step:
1. `foo` calls `bar`.
2. `bar` calls `baz`.
Therefore, `foo` calls bar indirectly. FINAL ANSWER: YES
Q: does method `bar` call method `foo`, either directly or indirectly?
A:
Let's think step by step:
1. `bar` calls `baz`.
2. `baz` does not call anything.
Therefore, `bar` does not call `foo`. FINAL ANSWER: NO.

Now we will have the real java code:
```
""",
    "end": """
```
Now the user will ask their question. Remember to think step by step and finish with FINAL ANSWER: YES or FINAL ANSWER: NO.
Q:
"""
}

in_context_tree_calls = {
    "start": """You are an assistant answering questions on source code, doing source code analysis. The questions will be about whether methods are calling each other, either directly or indirectly. To answer questions, think step-by-step by following method calls. Before answering with YES or NO, you must explain your reasoning step by step.
Be truthful, I don’t care whether the methods call each other or not, it does not affect me. Always end your answer with FINAL ANSWER: YES or FINAL ANSWER: NO. Here is an example code snippet:
```
public class Example {
    public void grault() {
        // End of chain
    }
    
    public void foo() {
        bar();
        baz();
    }
    
    public void corge() {
        // End of chain
    }
    
    public void baz() {
        quux();
        grault();
    }

    public void qux() {
        // End of chain
    }
        
    public void bar() {
        qux();
        corge();
    }
    
    public void quux() {
        // End of chain
    }
        
}```
Here are example questions:
    
Q: does method `foo` call method `corge`, either directly or indirectly?
A:
Let's think step by step:
1. `foo` calls `bar`.
2. `bar` calls `qux`.
3. `qux` does not call anything.
3. `bar` calls `corge`.
Therefore, `foo` calls `corge` indirectly. FINAL ANSWER: YES
    
Q: does method `bar` call method `grault`, either directly or indirectly?
A:
Let's think step by step:
1. `bar` calls `qux`.
2. `qux` does not call anything.
3. `bar` calls `corge`.
4. `corge` does not call anything.
Therefore, `bar` does not call `grault`. FINAL ANSWER: NO.

Now we will have the real java code:
``` 
""",
    "end": """
```
Now the user will ask their question. Remember to think step by step and finish with FINAL ANSWER: YES or FINAL ANSWER: NO.
Q:
"""
}

in_context_control_flow_1_tree_calls = {
    "start": """You are an assistant answering questions on source code, doing source code analysis. The questions will be about whether methods are calling each other, either directly or indirectly. To answer questions, think step-by-step by following method calls. Before answering with YES or NO, you must explain your reasoning step by step.
Be truthful, I don’t care whether the methods call each other or not, it does not affect me. Always end your answer with FINAL ANSWER: YES or FINAL ANSWER: NO. Here is an example code snippet:
```
public class Example {
    public void grault() {
        double height = 7.25;
        int counter = 0;
        while (counter < 4) {
            System.out.println(counter);
			counter++;
        }
        if (height >= 3.57) {
			System.out.println(height);
        }
        // End of chain
    }
    
    public void foo() {
        boolean flag = true;
        if (flag != false) {
            bar();
        }
        for (int i = 0; i < 7; i++) {
            baz();
        }
    }
    
    public void corge() {
        int error = 8;
		for (int i = 0; i < 5; i++) {
			System.out.println(i);
		}
		if (error >= 5) {
			System.out.println(error);
		}
        // End of chain
    }
    
    public void baz() {
        int length = 3;
		if (length <= 8) {
            quux();
		}
		int counter = 0;
		while (counter < 5) {
            grault();
			counter++;
		}
    }

    public void qux() {
        int buffer = 6;
		for (int i = 0; i < 1; i++) {
			System.out.println(i);
		}
		if (buffer >= 2) {
			System.out.println(buffer);
		}
		// End of chain
    }
        
    public void bar() {
        long line = 6;
        for (int i = 0; i < 2; i++) {
            qux();
        }
        if (line >= 3) {
            corge();
        }
    }
    
    public void quux() {
        double z = 4.75;
		int counter = 0;
		while (counter < 3) {
			System.out.println(counter);
			counter++;
		}
		if (z <= 6.59) {
			System.out.println(z);
		}
		// End of chain
    }
}```
Here are example questions:
Q: does method `foo` call method `baz`, either directly or indirectly?
A:
Let's think step by step:
1. `foo` calls `bar`.
2. `bar` calls `baz`.
Therefore, `foo` calls bar indirectly. FINAL ANSWER: YES
Q: does method `bar` call method `foo`, either directly or indirectly?
A:
Let's think step by step:
1. `bar` calls `baz`.
2. `baz` does not call anything.
Therefore, `bar` does not call `foo`. FINAL ANSWER: NO.

Now we will have the real java code:
```
""",
    "end": """
```
Now the user will ask their question. Remember to think step by step and finish with FINAL ANSWER: YES or FINAL ANSWER: NO.
Q:
"""
}

in_context_control_flow_2_tree_calls = {
    "start": """You are an assistant answering questions on source code, doing source code analysis. The questions will be about whether methods are calling each other, either directly or indirectly. To answer questions, think step-by-step by following method calls. Before answering with YES or NO, you must explain your reasoning step by step.
Be truthful, I don’t care whether the methods call each other or not, it does not affect me. Always end your answer with FINAL ANSWER: YES or FINAL ANSWER: NO. Here is an example code snippet:
```
public class Example {
    public void grault() {
        double height = 7.25;
		boolean temp = true;
        int counter = 0;
        while (counter < 4) {
            System.out.println(counter);
			counter++;
        }
        if (height >= 3.57) {
			System.out.println(height);
        }
        for (int i = 0; i < 5; i++) {
			System.out.println(i);
		}
		if (temp == true) {
			System.out.println(height);
		}
        // End of chain
    }
    
    public void foo() {
        boolean flag = true;
		long z = 7;
        if (flag != false) {
            bar();
        }
        for (int i = 0; i < 7; i++) {
			System.out.println(i);
        }
        for (int i = 0; i < 4; i++) {
            baz();
		}
		if (z >= 6) {
			System.out.println(flag);
        }
    }
    
    public void corge() {
        int error = 8;
		int index = 6;
		for (int i = 0; i < 5; i++) {
			System.out.println(i);
		}
		if (error >= 5) {
			System.out.println(error);
		}
        if (index >= 5) {
			System.out.println(index);
		}
		int counter = 0;
		while (counter < 5) {
			System.out.println(counter);
			counter++;
		}
        // End of chain
    }
    
    public void baz() {
		int status = 10;
        int length = 3;
		if (length <= 8) {
			System.out.println(status);
        }
		int counter = 0;
		while (counter < 5) {
			System.out.println(counter);
			counter++;
		}
		counter = 0;
		while (counter < 5) {
            quux();
			counter++;
		}
        if (status >= 9) {
            grault();
		}
    }

    public void qux() {
		boolean node = true;
        int buffer = 6;
		for (int i = 0; i < 1; i++) {
			System.out.println(i);
		}
		for (int i = 0; i < 3; i++) {
			System.out.println(i);
		}
		if (buffer >= 2) {
			System.out.println(buffer);
		}
        if (node == true) {
			System.out.println(buffer);
		}
		// End of chain
    }
        
    public void bar() {
		long message = 5;
        long line = 6;
        for (int i = 0; i < 2; i++) {
            qux();
        }
        if (line >= 3) {
			System.out.println(line);
        }
		if (message <= 7) {
            corge();
		}
        int counter = 0;
		while (counter < 2) {
			System.out.println(message);
			counter++;
		}
    }
    
    public void quux() {
        double z = 4.75;
		long path = 6;
		if (z <= 6.59) {
			System.out.println(z);
		}
        if (path >= 5) {
			System.out.println(path);
		}
		int counter = 0;
		while (counter < 3) {
			System.out.println(counter);
			counter++;
		}
		counter = 0;
		while (counter < 1) {
			System.out.println(counter);
			counter++;
		}
		// End of chain
    }
}```
Here are example questions:
Q: does method `foo` call method `baz`, either directly or indirectly?
A:
Let's think step by step:
1. `foo` calls `bar`.
2. `bar` calls `baz`.
Therefore, `foo` calls bar indirectly. FINAL ANSWER: YES
Q: does method `bar` call method `foo`, either directly or indirectly?
A:
Let's think step by step:
1. `bar` calls `baz`.
2. `baz` does not call anything.
Therefore, `bar` does not call `foo`. FINAL ANSWER: NO.

Now we will have the real java code:
```
""",
    "end": """
```
Now the user will ask their question. Remember to think step by step and finish with FINAL ANSWER: YES or FINAL ANSWER: NO.
Q:
"""
}