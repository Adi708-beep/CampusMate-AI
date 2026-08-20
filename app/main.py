from app.rag import CollegeRAG


def main():

    rag = CollegeRAG()

    print("\n==============================")
    print("        CAMPUSMATE AI")
    print("==============================")
    print("Type 'exit' to stop.\n")

    while True:

        question = input("You: ")

        if question.lower() == "exit":
            print("\nGoodbye!")
            break

        try:

            answer, sources = rag.ask(question)

            print("\nCampusMate:")
            print(answer)

            print("\nSources:")

            for source in sources:
                print("-", source)

            print("\n" + "-" * 40)

        except Exception as e:

            print("\nError:", e)


if __name__ == "__main__":
    main()