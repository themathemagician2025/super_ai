import { MathematicalConjecture } from '../conjectures/MathematicalConjecture';

export class AIEngine {
    private conjectures: MathematicalConjecture[];

    constructor(conjectures: MathematicalConjecture[]) {
        this.conjectures = conjectures;
    }

    analyze(input: number[]): number {
        return this.conjectures.reduce((acc, conjecture) => {
            return acc + conjecture.evaluate(input) * conjecture.confidence();
        }, 0) / this.conjectures.length;
    }

    addConjecture(conjecture: MathematicalConjecture) {
        this.conjectures.push(conjecture);
    }
}
