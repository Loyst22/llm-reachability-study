package reachability_test_ideas.control_flow;

public class useless_control_flow {
    public void function1() {
        int x = 0;
        int y;
        if (x == 0) {
            y = 1;
        } else {
            y = 2;
        }
        if (y == 1) {
            y = 2;
        } else {
            y = 3;
        }
        function2();
    }

    public void function2() {
        int z = 0;
        if (z == 0) {
            z = 1;
        } else {
            z = 2;
        }
        function3();
    }

    public void function3() {
        int cpt = 0;
        if (cpt == 0) {
            cpt = 1;
        } else {
            cpt = 2;
        }
    }

    public void function4() {
        int var = 0;
        if (var == 0) {
            var = 1;
        } else {
            var = 2;
        }
        function5();
    }

    public void function5() {
        int c = 0;
        if (c == 0) {
            c = 1;
        } else {
            c = 2;
        }
        function6();
    }

    public void function6() {
        int d = 0;
        if (d == 0) {
            d = 1;
        } else {
            d = 2;
        }
    }
}
