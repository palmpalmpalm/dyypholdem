const RANK_NAMES: Record<string, string> = {
  '2': 'Two',
  '3': 'Three',
  '4': 'Four',
  '5': 'Five',
  '6': 'Six',
  '7': 'Seven',
  '8': 'Eight',
  '9': 'Nine',
  T: 'Ten',
  J: 'Jack',
  Q: 'Queen',
  K: 'King',
  A: 'Ace',
}

const SUITS: Record<string, { glyph: string; name: string; red: boolean }> = {
  c: { glyph: '♣', name: 'clubs', red: false },
  d: { glyph: '♦', name: 'diamonds', red: true },
  h: { glyph: '♥', name: 'hearts', red: true },
  s: { glyph: '♠', name: 'spades', red: false },
}

export function formatChips(value: number | null | undefined): string {
  return Math.round(value ?? 0).toLocaleString('en-US')
}

export function formatSignedChips(value: number): string {
  return `${value >= 0 ? '+' : '−'}${formatChips(Math.abs(value))}`
}

export function formatSeconds(value: number | null | undefined, digits = 3): string {
  return value === null || value === undefined ? '—' : `${value.toFixed(digits)}s`
}

export function cardPresentation(card: string): {
  rank: string
  suit: string
  red: boolean
  label: string
} | null {
  const rank = RANK_NAMES[card[0]]
  const suit = SUITS[card[1]]
  if (!rank || !suit) return null
  return {
    rank: card[0],
    suit: suit.glyph,
    red: suit.red,
    label: `${rank} of ${suit.name}`,
  }
}
