import time

def binary_divisible_by_5():
    print("=== DFA: Podzielność przez 5 (System Binarny) ===")
    binary_input = input("Podaj ciąg binarny (np. 1010 dla 10): ").strip()

    transitions = {
        0: {0: 0, 1: 1}, # (0*2+0)%5=0, (0*2+1)%5=1
        1: {0: 2, 1: 3}, # (1*2+0)%5=2, (1*2+1)%5=3
        2: {0: 4, 1: 0}, # (2*2+0)%5=4, (2*2+1)%5=0
        3: {0: 1, 1: 2}, # (3*2+0)%5=1, (3*2+1)%5=2
        4: {0: 3, 1: 4}, # (4*2+0)%5=3, (4*2+1)%5=4
    }

    current_state = 0
    decimal_value = 0
    print("\nProces analizy:")
    print(f"{'Bit':^5} | {'Wartość (dec)':^15} | {'Przejście (Stan)':^20}")
    print("-" * 45)

    for bit_char in binary_input:
        if bit_char not in ('0', '1'):
            print(f"BŁĄD: '{bit_char}' to nie bit!")
            return
        bit = int(bit_char)
        old_state = current_state
        decimal_value = decimal_value * 2 + bit
        current_state = transitions[current_state][bit]
        print(f"{bit_char:^5} | {decimal_value:^15} | S{old_state} --({bit_char})--> S{current_state}")
        time.sleep(0.5)

    print("-" * 45)
    if current_state == 0:
        print(f"WYNIK: Liczba {decimal_value} jest PODZIELNA przez 5. (Stan końcowy S0)")
    else:
        print(f"WYNIK: Liczba {decimal_value} NIE jest podzielna przez 5. (Reszta = {current_state})")
binary_divisible_by_5()