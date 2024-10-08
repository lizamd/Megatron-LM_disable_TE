filename = "output.txt"

with open(filename, "r") as file:
    for line in file:
        parts = line.split()
        if len(parts) > 1 and parts[0].startswith("HIPPerfBufferCopySpeed"):
            print(parts[-1])
