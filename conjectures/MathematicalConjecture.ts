export interface MathematicalConjecture {
    name: string;
    evaluate(input: number[]): number;
    confidence(): number;
}
