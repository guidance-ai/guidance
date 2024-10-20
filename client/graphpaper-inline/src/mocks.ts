// Mocks for interactive testing

import {type TextOutput, type RoleOpenerInput, type RoleCloserInput, type GenToken} from './stitch';

export const mockNodeAttrs: Array<RoleCloserInput | RoleOpenerInput | TextOutput> = []
export const mockGenTokens: Array<GenToken> = [];
