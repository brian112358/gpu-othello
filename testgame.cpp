#include <string>
#include <cstdlib>
using namespace std;

int main(int argc, char *argv[]) {
    // Invoke the Java program with the passed arguments.
    string cmd = "java -cp OthelloFramework.jar TestGame";    
    argv++;
    while (--argc) {
        cmd += " ";
        cmd += *(argv++);
    }
    system(cmd.c_str());
    return 0;
}
