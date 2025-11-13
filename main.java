class Person {
    String name;
    int age;
}
class Student extends Person {
    int rollNo;
}
class Teacher extends Student {
    String subjectCode;
}
public class Main {
public static void main(String[] args) {
Teacher t = new Teacher();
t.name = "Ankit";
t.age = 21;
t.rollNo = 10926;
t.subjectCode = "CS101";
System.out.println("Name: " + t.name);
System.out.println("Age: " + t.age);
System.out.println("Roll No: " + t.rollNo);
System.out.println("Subject Code: " + t.subjectCode);
}
}