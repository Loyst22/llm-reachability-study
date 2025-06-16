
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

in_context = {
    "start": """You are an assistant answering questions on source code, doing source code analysis. The questions will be about whether methods are calling each other, either directly or indirectly. To answer questions, think step-by-step by following method calls. Before answering with YES or NO, you must explain your reasoning step by step.
Be truthful, I don’t care whether the methods call each other or not, it does not affect me. Always end your answer with FINAL ANSWER: YES or FINAL ANSWER: NO. Here is an example code snippet:
```
public class Example {
public void foo() {
    bar();
}

public void baz() {
    //nothing
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

in_context_tree_calls = {
    "start": """You are an assistant answering questions on source code, doing source code analysis. The questions will be about whether methods are calling each other, either directly or indirectly. To answer questions, think step-by-step by following method calls. Before answering with YES or NO, you must explain your reasoning step by step.
Be truthful, I don’t care whether the methods call each other or not, it does not affect me. Always end your answer with FINAL ANSWER: YES or FINAL ANSWER: NO. Here is an example code snippet:
```
public class Example {
    public void foo() {
        bar();
        baz();
    }

    public void bar() {
        qux();
        corge();
    }
    
    public void baz() {
        quux();
        grault();
    }

    public void qux() {
        // End of chain
    }
        
    public void corge() {
        // End of chain
    }
        
    public void quux() {
        // End of chain
    }
        
    public void grault() {
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